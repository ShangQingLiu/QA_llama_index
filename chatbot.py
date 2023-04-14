from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from gpt_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, GraphToolConfig
from gpt_index.indices.query.query_transform.base import DecomposeQueryTransform
from gpt_index import LLMPredictor
from langchain import OpenAI


class ChatBot():
    def __init__(self,index_set,graph=None) -> None:
        self.get_query_configs = self.get_query_configs()
        self.index_configs = self.getIndexConfigs(index_set)
        self.graph_config =  None if graph == None else self.getGraphConfig(graph)
        self.toolKit = self.getToolKit()
        self.agent = self.getAgent(self.toolKit) 

    def run(self,query)->str:
        response = self.agent.run(input=query)
        return response 

        # define query configs for graph 
    def get_query_configs(self):
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
        decompose_transform = DecomposeQueryTransform(
            llm_predictor, verbose=True
        )
        query_configs = [
            {
                "index_struct_type": "simple_dict",
                "query_mode": "default",
                "query_kwargs": {
                    "similarity_top_k": 1,
                    # "include_summary": True
                },
                "query_transform": decompose_transform
            },
            {
                "index_struct_type": "list",
                "query_mode": "default",
                "query_kwargs": {
                    "response_mode": "tree_summarize",
                    "verbose": True
                }
            },
        ]
        return query_configs
    
    def getToolKit(self):
        toolkit = None

        if self.graph_config == None:
            toolkit = LlamaToolkit(
                index_configs=self.index_configs,
            )
        else:
            toolkit = LlamaToolkit(
                index_configs=self.index_configs,
                graph_configs=[self.graph_config]
            ) 

        return toolkit
    
    def getIndexConfigs(index_set):
        index_configs = []

        for key in index_set.keys():
            tool_config = IndexToolConfig(
                index=index_set[key], 
                name=f"Vector Index {key}",
                description=f"useful for when you want to answer queries about the {key} SEC 10-K for Uber",
                index_query_kwargs={"similarity_top_k": 3},
                tool_kwargs={"return_direct": True}
            )
            index_configs.append(tool_config)

        return index_configs

    def getGraphConfig(self):
        graph_config = GraphToolConfig(
            graph = self.graph,
            name="Graph Index",
            description="useful for when you want to answer queries about the SEC 10-K for Uber",
            index_query_kwargs={"similarity_top_k": 3},
            tool_kwargs={"return_direct": True}
        )

        return graph_config

    def getAgent(self,toolkit):
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm=OpenAI(temperature=0)
        agent = create_llama_chat_agent( toolkit,
            llm,
            memory=memory,
            verbose=True
        )
        return agent