import os


def set_langsmith():
    # LangSmith tracing deshabilitado — causaba rate limit 429 y ruido en logs
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
