import os
import openai
from enum import Enum

from gpt_index import download_loader, GPTSimpleVectorIndex, ServiceContext
from pathlib import Path
from gpt_index import GPTListIndex, LLMPredictor
from langchain import OpenAI
from gpt_index.indices.composability import ComposableGraph

class DataType(Enum):
    AUDIO = 1
    DOCX = 2
    PDF = 3
    HTML = 4
    # IMAGE = 5

    
class IndexUtils():
    def __init__(self, root_path, project_name="default"):
        self.root_path = root_path # index saved root path
        self.project_name = project_name # would later used as graph index name
        
    # Load data and return index_sets
    def dataLoader(self, file_pathes: list, data_type: DataType): 
        reader = None
        if data_type == DataType.HTML:
            UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
            reader = UnstructuredReader()
        elif data_type == DataType.DOCX: 
            DocxReader = download_loader("DocxReader")
            reader = DocxReader()
        elif data_type == DataType.PDF:
            #TODO: not tested
            PDFReader = download_loader("PDFReader")
            reader = PDFReader()
        elif data_type == DataType.AUDIO:
            #TODO: not tested 
            AudioTranscriber = download_loader("AudioTranscriber")
            reader = AudioTranscriber()

        if reader is None:
            raise ValueError("The data type is not supported!")
        

        unsaved_doc_set = {}
        saved_doc_path = [] 
        for file_path_str in file_pathes:
            file_name = os.path.basename(file_path_str)
            # check if index already exists
            index_name = "index_" + file_name
            index_path = os.path.join(self.root_path,index_name)
            if os.path.exists(index_path):
                saved_doc_path.append(index_path)
                continue 

            unsaved_doc_set[file_name] = reader.load_data(Path(file_path_str),split_documents=False)

        index_set = {} 
        index_set.update(self.saveIndexer(512, unsaved_doc_set))
        index_set.update(self.loadIndexer(saved_doc_path))

        return index_set
    
    def saveIndexer(self, chunk_size_limit, doc_set):
        index_set = {}
        service_context = ServiceContext.from_defaults(chunk_size_limit=chunk_size_limit)
        for key in doc_set.keys():
            cur_index = GPTSimpleVectorIndex.from_documents(doc_set[key], service_context=service_context)
            index_set[key] = cur_index
            cur_index.save_to_disk(os.path.join(self.root_path,f'index_{key}.json'))
        return index_set

    def loadIndexer(self, pathes:list):
        index_set = {}

        for path in pathes:
            cur_index = GPTSimpleVectorIndex.load_from_disk(Path(path))
            file_name = os.path.basename(path)
            index_set[file_name] = cur_index

        return index_set
    def buildGraphIndexer(self, indexers):
        # check if graph exists
        file_name = f"graph_{self.project_name}.json"
        graph_path = os.path.join(self.root_path,file_name) 

        if os.path.exists(graph_path):
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
            graph = ComposableGraph.load_from_disk(Path(graph_path),ServiceContext=service_context)

            return graph

        # set summary text for each doc
        # TODO: Now only use filename as summary text
        index_summaries = indexers.keys() 

        # set number of output tokens
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

        # define a list index over the vector indices
        # allows us to synthesize information across each index
        graph = ComposableGraph.from_indices(
            GPTListIndex, 
            [indexers[key] for key in indexers.keys()], 
            index_summaries=index_summaries,
            service_context=service_context
        )

        return graph
