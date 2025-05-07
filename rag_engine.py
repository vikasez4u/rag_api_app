import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from typing import Dict, Any

class RAGEngine:
    def __init__(self, base_dir: str = os.getcwd(), ollama_base_url: str = "http://localhost:11434"):
        
        self.base_dir = base_dir
        self.document_dir = os.path.join(base_dir, "DocumentDir")
        self.ollama_base_url = ollama_base_url
        self.index = None
        
        # Create document directory if it doesn't exist
        if not os.path.exists(self.document_dir):
            os.makedirs(self.document_dir)
            print(f"Created document directory: {self.document_dir}")
        
        # Setup Ollama embedding and LLM
        self._setup_ollama()
        
        # Setup prompt templates
        self._setup_prompts()
        
        
    
    def _setup_ollama(self):
        # Setup embedding model
        ollama_embedding = OllamaEmbedding(
            model_name="llama3",
            base_url=self.ollama_base_url,
            ollama_additional_kwargs={"mirostat": 0},
        )
        Settings.embed_model = ollama_embedding
        
        # Setup LLM
        Settings.llm = Ollama(model="llama3", base_url=self.ollama_base_url)
    
    def _setup_prompts(self):
        """Setup prompt templates for query and refinement"""
        # Text QA template
        text_qa_template_str = (
            "Context information is"
            " below.\n---------------------\n{context_str}\n---------------------\nUsing"
            " both the context information and also using your own knowledge, answer"
            " the question: {query_str}\nIf the context isn't helpful, you can also"
            " answer the question on your own.\n"
            " answer using facts but keep it clean and concise so that everyone can understand clearly"
            " ensure you understand the users query and ask follow up questions if required"
            " format the reponse and ensure it is presentable"
            " Create table structure where needed in the response"
        )
        self.text_qa_template = PromptTemplate(text_qa_template_str)
        
        # Refine template
        refine_template_str = (
            " You are an senior subject matter expert in the banking and finance domain"
            " your speciality is payments. The queries you will get will be related to payments"
            " Your users will be software developers, testers, product owners"
            " Users will need help with Acceptance Criteria Generation"
            " Test Design, Code review etc. Keeping the context in mind answer the question"
            " The original question is as follows:\n {query_str} \n We have provided an"
            " existing answer: {existing_answer}\n We have the opportunity to refine"
            " the existing answer meeting the corporate standards with some more context"
            " \n------------\n{context_msg}\n------------\n Using both the new"
            " context and your own knowledge, update or repeat the existing answer.\n"
            " ensure there is enough space above and below the query to maintain proper document format"
            " Be precise with the answer and ensure answer is in tabular format where needed"
        )
        self.refine_template = PromptTemplate(refine_template_str)
    
    def load_data(self) -> None:
        
        print(f"Loading documents from {self.document_dir}...")
        
        try:
            # Load documents
            documents = SimpleDirectoryReader(input_dir=self.document_dir).load_data()
            print(f"Loaded {len(documents)} documents")
            
            # Create index
            self.index = VectorStoreIndex.from_documents(documents)
            print("Index created successfully")
            
            return {"status": "success", "document_count": len(documents)}
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            raise
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        if not self.index:
            raise ValueError("Index not initialized. Call load_data() first.")
        
        # Create query engine with templates
        query_engine = self.index.as_query_engine(
            text_qa_template=self.text_qa_template,
            refine_template=self.refine_template,
            similarity_top_k=2
        )
        
        # Get response
        response = query_engine.query(question)
        
        # Format response with source information
        if hasattr(response, 'source_nodes') and len(response.source_nodes) > 0:
            if len(response.source_nodes) >= 2:
                file_name = (response.source_nodes[0].text + 
                            response.source_nodes[0].node.metadata['file_name'] + 
                            "page nos: " + response.source_nodes[0].node.metadata['page_label'] + 
                            ", " + response.source_nodes[1].node.metadata['page_label'])
            else:
                file_name = (response.source_nodes[0].text + 
                            response.source_nodes[0].node.metadata['file_name'] + 
                            "page nos: " + response.source_nodes[0].node.metadata['page_label'])
            
            final_response = response.response + '\n\n Check further at ' + file_name
        else:
            final_response = response.response
        
        # Extract source nodes for API response
        source_documents = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                metadata = node.node.metadata if hasattr(node, 'node') else {}
                source_documents.append({
                    'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    'score': float(node.score) if hasattr(node, 'score') else None,
                    'file_name': metadata.get('file_name', 'Unknown'),
                    'page_label': metadata.get('page_label', 'Unknown')
                })
        
        # Format API response
        result = {
            'question': question,
            'answer': final_response,
            'raw_answer': str(response.response),
            'sources': source_documents
        }
        
        return result