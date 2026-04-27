from rag.config import EXPERIMENTS
from rag.pipeline import RAGPipeline
from rag.client import RAGClient


def default_rag_client(llm_client, experiment="baseline", logdir=None):
    config = EXPERIMENTS[experiment]

    pipeline = RAGPipeline(
        llm_client=llm_client,
        config=config,
        experiment_name=experiment,
    )

    return RAGClient(
        pipeline=pipeline,
        experiment_name=experiment,
        logdir=logdir,
    )