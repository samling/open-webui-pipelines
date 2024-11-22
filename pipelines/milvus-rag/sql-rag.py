"""
title: SQL RAG Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a database using the Haystack library.
requirements: haystack-ai,pandas
"""
import datetime
import os
import pandas as pd
import sqlite3
from typing import List, Optional, Union, Generator, Iterator
from pydantic import BaseModel
import os

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str
        OPENAI_BASE_URL: str
        PROMPT_BASE_MODEL: str
        PROMPT_ANALYSIS_MODEL: str
        DB_PATH: str

    def __init__(self):
        self.sql_pipeline = None
        self.valves = self.Valves(
            **{
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "sk-fake-key"),
                "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "http://openai"),
                "PROMPT_BASE_MODEL": os.getenv("PROMPT_BASE_MODEL", "gpt-4o-2024-11-20"),
                "PROMPT_ANALYSIS_MODEL": os.getenv("PROMPT_ANALYSIS_MODEL", "gpt-4o-mini"),
                "DB_PATH": os.getenv("DB_PATH", "")
            }
        )

    async def on_startup(self):
        os.environ["OPENAI_API_KEY"] = self.valves.OPENAI_API_KEY
        os.environ['OPENAI_BASE_URL'] = self.valves.OPENAI_BASE_URL

        from haystack import component, Pipeline
        from haystack.components.builders import PromptBuilder
        from haystack.components.generators import OpenAIGenerator

        @component
        class QueryHelper:
            def __init__(self, sql_database: str):
                self.db_path = sql_database
                self._connection = None
                print(f"QueryHelper initialized with db_patch: {self.db_path}")
                if not os.path.exists(self.db_path):
                    print(f"ERROR: Database file does not exist at {self.db_path}")
                else:
                    print(f"Database file found at {self.db_path}")

            @property
            def connection(self):
                if self._connection is None:
                    try:
                        print(f"Attempting to connect to database at {self.db_path}")
                        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
                        print(f"Successfully connected to database")
                    except sqlite3.Error as e:
                        print(f"SQLite connection error: {str(e)}")
                    except Exception as e:
                        print(f"Unexpected error connect to the database: {str(e)}")
                return self._connection

            def get_current_date(self):
                now = datetime.datetime.now()
                curr_date = now.strftime("%Y-%m-%d")
                return curr_date

            def get_schema(self, table_name: str) -> str:
                # Get column information using SQLAlchemy inspector
                cursor = self.connection.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Format column information
                schema_info = []
                for col in columns:
                    schema_info.append(f"{col[1]} ({col[2]})")
                
                return ", ".join(schema_info)

            def get_categories(self, table_name: str) -> List[str]:
                try:
                    cursor = self.connection.cursor()
                    cursor.execute(f"SELECT DISTINCT category FROM {table_name} WHERE category IS NOT NULL ORDER BY category")
                    categories = cursor.fetchall()
                    return ", ".join([f"{category[0]} " for category in categories])
                except Exception as e:
                    return ""

            def get_accounts(self, table_name: str) -> List[str]:
                try:
                    cursor = self.connection.cursor()
                    cursor.execute(f"SELECT DISTINCT account FROM {table_name} WHERE account IS NOT NULL ORDER BY account")
                    accounts = cursor.fetchall()
                    return ", ".join([f"{account[0]} " for account in accounts])
                except Exception as e:
                    return ""

            @component.output_types(curr_date=str, schema=str, categories=str, accounts=str)
            def run(self, table_name: str = 'transactions'):
                curr_date = self.get_current_date()
                schema = self.get_schema(table_name)
                categories = self.get_categories(table_name)
                accounts = self.get_accounts(table_name)
                return {"curr_date": curr_date, "schema": schema, "categories": categories, "accounts": accounts}

        @component
        class SQLQuery:
            def __init__(self, sql_database: str):
                self.db_path = sql_database
                self._connection = None
                print(f"QueryHelper initialized with db_patch: {self.db_path}")
                if not os.path.exists(self.db_path):
                    print(f"ERROR: Database file does not exist at {self.db_path}")
                else:
                    print(f"Database file found at {self.db_path}")

            def _get_connection(self):
                if self._connection is None:
                    try:
                        print(f"Attempting to connect to database at {self.db_path}")
                        self._connection = sqlite3.connect(
                            self.db_path,
                            check_same_thread=False,
                            timeout=10.0,
                            isolation_level=None
                        )
                        print("Successfully connected to database")
                        self._connection.execute("PRAGMA journal_mode=WAL")
                        self._connection.execute("PRAGMA synchronous=NORMAL")
                        self._connection.execute("PRAGMA cache_size=2000") # 2mb cache
                        self._connection.execute("PRAGMA temp_store=MEMORY")
                    except sqlite3.Error as e:
                        print(f"SQLite connection error: {str(e)}")
                    except Exception as e:
                        print(f"Unexpected error connect to the database: {str(e)}")
                return self._connection

            @component.output_types(results=Optional[str], query=str, error_message=Optional[str])
            def run(self, queries: List[str]):
                query = queries[0]
                connection = self._get_connection()
                try:
                    df = pd.read_sql_query(
                        query,
                        connection,
                    )
                    result = df.to_string(index=False)
                    return {"results": result, "query": query, "error_message": None}
                except Exception as e:
                    error_message = f"Database query error: {str(e)}"
                    return {"results": None, "query": query, "error_message": error_message}

        sql_prompt = PromptBuilder(template="""
                    Please generate a SQL query. The query should answer the following Question: {{question}};

                    The current date is:
                    {{curr_date}}

                    The table 'transactions' has the following schema:
                    {{schema}}

                    The available categories are:
                    {{categories}}

                    The available accounts are:
                    {{accounts}}

                    Follow these suggestions for building a better query:
                        - Be creative with your answer. For example, a query asking for transactions by a certain merchant name may have
                        the merchant name in the merchant column, the description column, and/or the account column.
                        - Use case-insensitive matching.
                        - Try alternative names; for example, if you know that SCE is Southern California Edison, look for instances of "Edison".
                        - Keep the query simple; avoid too many AND statements.
                        - Keep the original column names in your query, only add additional columns to the result if necessary.
                        - Order the results from least to most recent.
                        - Avoid relying on the category column; you can use it, but often times transactions are miscategorized or missing a category altogether.

                    You should only respond with the SQL query without any code formatting, including no triple backticks or ```.

                    Answer:""")
        analysis_prompt = PromptBuilder(template="""I asked: {{question}}

                    The SQL query used was:
                    {{query}}

                    {% if error_message %}

                    There was an error executing the query:

                    {{error_message}}
                    
                    Inform the user there was an error and helpfully and briefly suggest that they try asking again, or rephrasing their question.
                    Let them know that sometimes models make mistakes and that rerunning can sometimes provide better results.

                    {% else %}

                    The data from the database shows:
                    {{results}}

                    Follow these instructions in formulating your response:
                        - If there are zero results, ignore the remaining instructions and inform the user there were no results.
                        Helpfully and briefly suggest that they try their search again, optionally tweaking the question for better results.
                        Do not include anything else other than this answer, specifically do not include the table.
                        - If there are more than zero results, print the results in a tabular format wrapped in a codeblock above your analysis.
                        - Start your answer with the table; don't tell the user anything like 'here is the result'.
                        - Translate any month numbers to names, and add a $ to any dollar amount.
                        - Don't truncate the list of results unless it exceeds 30 rows.
                        - Add a natural language analysis of this data that answers my original question. 
                        - Include specific numbers and trends if relevant. 
                        - Make it conversational but informative.

                    {% endif %}

                    Response:""")

        query_helper = QueryHelper(sql_database=self.valves.DB_PATH)
        sql_generator = OpenAIGenerator(model=self.valves.PROMPT_BASE_MODEL, timeout=30)
        sql_querier = SQLQuery(sql_database=self.valves.DB_PATH)
        analysis_generator = OpenAIGenerator(model=self.valves.PROMPT_ANALYSIS_MODEL, timeout=30)

        self.sql_pipeline = Pipeline()

        # Create components
        self.sql_pipeline.add_component("query_helper", query_helper)
        self.sql_pipeline.add_component("sql_prompt", sql_prompt)
        self.sql_pipeline.add_component("sql_generator", sql_generator)
        self.sql_pipeline.add_component("sql_querier", sql_querier)
        self.sql_pipeline.add_component("analysis_prompt", analysis_prompt)
        self.sql_pipeline.add_component("analysis_generator", analysis_generator)

        # Load the initial prompt with the query and additional context
        self.sql_pipeline.connect("query_helper.curr_date", "sql_prompt.curr_date")
        self.sql_pipeline.connect("query_helper.schema", "sql_prompt.schema")
        self.sql_pipeline.connect("query_helper.categories", "sql_prompt.categories")
        self.sql_pipeline.connect("query_helper.accounts", "sql_prompt.accounts")

        # Send the prompt to the generator
        self.sql_pipeline.connect("sql_prompt.prompt", "sql_generator.prompt")
        
        # Send the reply to the SQL querier
        self.sql_pipeline.connect("sql_generator.replies", "sql_querier.queries")
        
        # Send the origin question, query, and results to the analyzer prompt
        self.sql_pipeline.connect("sql_querier.results", "analysis_prompt.results")
        self.sql_pipeline.connect("sql_querier.query", "analysis_prompt.query")
        self.sql_pipeline.connect("sql_querier.error_message", "analysis_prompt.error_message")

        # Send the prompt to the analyzer
        self.sql_pipeline.connect("analysis_prompt.prompt", "analysis_generator.prompt")

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.


        question = user_message

        try:
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
            result = self.sql_pipeline.run(
                {
                    "query_helper": {
                        "table_name": "transactions",
                    },
                    "sql_prompt": {
                        "question": question
                    },
                    "analysis_prompt": {
                        "question": question
                    }
                }
            )

            if "analysis_generator" in result and result["analysis_generator"]["replies"]:
                return result['analysis_generator']['replies'][0]
            elif "sql_querier" in result and result["sql_querier"]["error_message"]:
                return result['sql_querier']['error_message']
            else:
                return "No analysis generated."

        except Exception as e:
            return f"Pipeline Error: {str(e)}"