import logging

from parameters import Parameters
from guesser import Guesser
from gpr_guesser import kINSTRUCTIONS

import dspy

class OllamaGuesser(Guesser):
    def __init__(self):        
        self._model = None
    
    def set_ollama_model(self, model, instructions=kINSTRUCTIONS):
        self._model = model

        self._instructions = instructions

    def query(self, question):
        from ollama import chat
        from ollama import ChatResponse
        
        assert self._model is not None, "You need to set the model first"

        response: ChatResponse = chat(model='gemma3:4b', messages=[
            {
                'role': 'user',
                'content': self._instructions  + "\n\nQuestion %s" % question,
            },
        ])
        
        return response

def validate_answer(example: dspy.Example, pred, trace=None):
    from eval import rough_compare

    correct = rough_compare(example.answer, pred.guess)

    retrieval_bonus = 0.0

    # This part should provide a signal to the query generation
    contexts = []
    for retrieved in pred.context.split("\n"):
        if retrieved.count(":") >= 2:
            guess, retriever, text = retrieved.split(":", 2)
            context_clear = {"guess": guess, "retriever": retriever, "text": text}            
            if rough_compare(guess, example.answer):
                retrieval_bonus += 1
            contexts.append(context_clear)
        
    # Normalize by number of guessers from tf-idf
    if len(contexts) > 0:
        retrieval_bonus /= len(contexts)

    logging.info("Retrieval bonus = %0.2f from  ans=%s clues=%s" % (retrieval_bonus, example.answer, ", ".join(x["guess"] for x in contexts)))
        
    # Bonus for correct confidence 
    if correct:
        return 1 + pred.confidence ** 2 + retrieval_bonus
    else:
        return - 3 * (pred.confidence ** 2)

   
class CalibratedRAG(dspy.Module):
    def __init__(self):

        class QueryGenerator(dspy.Signature):
            question: str = dspy.InputField()
            query: str = dspy.OutputField()

        class PredictionGenerator(dspy.Signature):
            question: str = dspy.InputField()
            query: str = dspy.InputField()
            context: str = dspy.InputField()
            guess: str = dspy.InputField()
            confidence: float = dspy.OutputField()

        self.topk = None
        self.retrievers = {}
        self.query_generator = dspy.Predict(QueryGenerator)
        self.guess_generator = dspy.ChainOfThought("question,context->answer")
        self.confidence_generator = dspy.Predict(PredictionGenerator)

    def forward(self, question, **kwargs):
        class FullResult(dspy.Signature):
            guess: str = dspy.OutputField()
            confidence: float = dspy.OutputField()
            context: str = dspy.OutputField()
        
        query = self.query_generator(question=question).query
        context = ""
        for retriever in self.retrievers:
            results = self.retrievers[retriever](query, self._topk)
            for result in results:
                context += "%s: %s: %s\n" % (result["guess"], retriever, result["question"])
            
        guess = self.guess_generator(question=question, context=context)
        confidence = self.confidence_generator(question=question, query=query, context=context, guess=guess)
        return FullResult(guess=guess.answer, context=context, confidence=confidence.confidence)

    def init_retriever(self, name:str, model_filename:str, topk:int=1):
        """
        Add a new retriever to the system.

        Parameters:
        - name: The key of the retriever
        - filename: Where to load the retriever from (filename)
        - topk: How many retrieved results to get from the retriever per query
        """
        
        logging.info("Loading retriever %s as %s (top k=%i)" % (model_filename, name, topk))  
        from tfidf_guesser import TfidfGuesser 
        retriever = TfidfGuesser(model_filename)
        self.retrievers[name] = retriever 
        self.retrievers[name].load()
        if self.topk is None:    
            self._topk = topk    
        else:                    
            assert self._topk == topk 
    
class OllamaDspy(Guesser):
    def __init__(self, params: Parameters):
        self.filename = params.ollama_guesser_filename
        self._qa_metric = validate_answer
        self._confidence_metric = lambda x, y: (x - y) ** 2
        self._model = CalibratedRAG()
        self._optimized = None

        if params.ollama_guesser_opt_algorithm == 'MIPROv2':
            from dspy.teleprompt import MIPROv2
            self._teleprompter = MIPROv2(
                metric=validate_answer,
                num_threads=params.ollama_guesser_threads,
                auto='medium',
            )
        elif params.ollama_guesser_opt_algorithm == 'BootstrapFewShotWithRandomSearch':
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch
            self._teleprompter = BootstrapFewShotWithRandomSearch(
                metric=validate_answer,
                #max_labeled_demos=64,
                #max_rounds=1,
                #max_bootstrapped_demos=16,
                num_threads=params.ollama_guesser_threads)
            
    @staticmethod
    def create_dataset(questions, answers):
        return [dspy.Example(question=x, answer=y).with_inputs("question") for x, y in zip(questions, answers)]

    def setup_retrieval(self, name:str, filename:str, topk:int):
        """
        Add a new retriever to the system.

        Parameters:
        - name: The key of the retriever
        - filename: Where to load the retriever from (filename)
        - topk: How many retrieved results to get from the retriever per query
        """
        self._model.init_retriever(name, filename, topk)
    
    def train(self, training_data, answer_field='page', split_by_sentence=True,
                    min_length=-1, max_length=-1, remove_missing_pages=True):
        Guesser.train(self, training_data, answer_field, split_by_sentence, min_length,
                      max_length, remove_missing_pages)

        dataset = self.create_dataset(self.questions, self.answers)
        
        self._optimized = self._teleprompter.compile(
            self._model,
            trainset=dataset,
        )

    def __call__(self, question, max_n_guesses=1):
        if self._optimized == None:
            model = self._model
        else:
            model = self._optimized

        assert max_n_guesses == 1, "Can only give one guess"
        result = model(question)
        return [{"guess": result.guess, "confidence": result.confidence, "evidence": result.context}]
            
            
    def set_ollama_model(self, model):
        self._lm = dspy.LM('ollama_chat/%s' % model, api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=self._lm)

    def load(self):
        path = self.filename
        self._model.load("%s.json" % path)        

    def save(self):
        path = self.filename
        if self._optimized is None:
            self._model.save("%s.json" % path)
        else:
            self._optimized.save("%s.json" % path)
        
if __name__ == "__main__":
    import argparse
    import gzip
    import json
    
    from guesser import kTOY_DATA

    from dspy.evaluate import Evaluate

    from nltk.tokenize import sent_tokenize

    from parameters import load_questions, add_guesser_params, add_general_params, add_question_params, setup_logging, instantiate_guesser

    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Question Type')
    add_general_params(parser)
    guesser_params = add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    guesser_params["ollama"].load_command_line_params(flags)

    questions = load_questions(flags)
    od = instantiate_guesser("Ollama", flags, False)

    # First, let's show some examples
    for qq in kTOY_DATA["train"]:
        result =  od(qq["text"])
        print(qq, result)

    # TODO: can we use some of the more standard question flags for this?
    with gzip.open(flags.ollama_guesser_dev_data) as infile:
        questions = json.load(infile)
        
        dev = od.create_dataset([sent_tokenize(x["text"])[:-1] for x in questions[:flags.ollama_guesser_validation_ex]],
                                [x["page"] for x in questions[:flags.ollama_guesser_validation_ex]])

        evaluator = Evaluate(devset=dev, num_threads=flags.ollama_guesser_threads,
                             display_progress=True, display_table=10, provide_traceback=True)

        print("PRETRAIN: %f" % evaluator(od._model, metric=validate_answer))

    
    od.train(questions[flags.ollama_guesser_validation_ex:])
    od.save()

    print("POSTTRAIN: %f" % evaluator(od._optimized, metric=validate_answer))
