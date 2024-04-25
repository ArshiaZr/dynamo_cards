from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from langchain_google_vertexai import VertexAI
from langchain.chains.summarize import load_summarize_chain
from vertexai.generative_models import GenerativeModel
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiProcessor:
    def __init__(self, model_name:str, project: str):
        self.model = VertexAI(model_name=model_name, project=project)

    
    def generate_document_summary(self, documents: list, **args):
        chain_type = 'map_reduce' if len(documents) > 10 else 'stuff'
        chain = load_summarize_chain(chain_type=chain_type, llm=self.model, **args)

        return chain.run(documents)
    
    def count_total_tokens(self, docs: list):
        temp_model = GenerativeModel("gemini-1.0-pro")
        total = 0
        logger.info("Counting total billable characters...")
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_billable_charachters
        return total
    
    def get_model(self):
        return self.model


class YoutubeProcessor:
    def __init__(self, genai_processor: GeminiProcessor):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.genAIProcessor = genai_processor

    def retrieve_youtube_documents(self, youtube_link: str, verbose = False):
        loader = YoutubeLoader.from_youtube_url(youtube_link, add_video_info=True)
        docs = loader.load()
        result = self.text_splitter.split_documents(docs)

        author = result[0].metadata["author"]
        length = result[0].metadata["length"]
        title = result[0].metadata["title"]
        total_size = len(result)

        if verbose:
            total_billable_characters = self.genAIProcessor.count_total_tokens(result)
            logging.info(f"{author}\n{length}\n{title}\n{total_size}\n{total_billable_characters}")

        return result
    
    def get_key_concepts(self, docs: list, sample_size: int=0, verbose=False):
        if sample_size > len(docs):
            raise ValueError("Group size is larger than the number of documents")
        
        if sample_size == 0:
            sample_size = len(docs) // 5
            if verbose:
                logger.info(f"Sample size not provided. Defaulting to {sample_size}")
        
        num_docs_per_group = len(docs) // sample_size + (len(docs) % sample_size > 0)

        
        if num_docs_per_group > 10:
            raise ValueError("Each group has more than 10 documents and output quality will degraded. Increase the sample_size parameter to reduce the number of documents per page.")
        elif num_docs_per_group > 5:
            logger.warn("Each group has more than 5 documents and output quality will be likely degraded. Consider increasing the sample_size parameter to reduce the number of documents per page.")

        groups = [docs[i: i + num_docs_per_group] for i in range(0, len(docs), num_docs_per_group)]

        concepts = []
        batch_cost = 0

        logger.info("Retreiving key concepts...")
        for group in tqdm(groups):
            content = ""

            for doc in group:
                content += doc.page_content
            
            prompt = PromptTemplate(
                template="""
                Find and define key concepts or terms found in the text:
                {text}
    
                Respond in the following format as a JSON object without any backticks separating each concept with a comma do not write ```json before the object or after the object:
                {{"concept1": "definition1", "concept2": "definition2", ...}}
                """,
                input_variables=["text"]
            )
            
            chain = prompt | self.genAIProcessor.model
            concept = chain.invoke({"text": content})
            concepts.append(concept)

            if verbose:
                total_input_char = len(content)
                total_input_cost = (total_input_char/1000) * 0.000125
                
                logging.info(f"Running chain on {len(group)} documents")
                logging.info(f"Total input characters: {total_input_char}")
                logging.info(f"Total cost: {total_input_cost}")
                
                total_output_char = len(concept)
                total_output_cost = (total_output_char/1000) * 0.000375
                
                logging.info(f"Total output characters: {total_output_char}")
                logging.info(f"Total cost: {total_output_cost}")
                
                batch_cost += total_input_cost + total_output_cost
                logging.info(f"Total group cost: {total_input_cost + total_output_cost}\n")
        

        # return concepts

        # remove ```json from the beginning and end of the string
        for i in range(len(concepts)):
            # if concepts[i].startswith("```json"):
            concepts[i] = concepts[i][7:]
            # if concepts[i].endswith("```"):
            concepts[i] = concepts[i][:-3]
        
        processed_concepts = [json.loads(concept) for concept in concepts]

        if verbose:
            logging.info(f"Total Analysis Cost: ${batch_cost}")
        return processed_concepts