import json
import os


class RAGClient:
    def __init__(self, pipeline, experiment_name, logdir=None):
        self.pipeline = pipeline
        self.experiment_name = experiment_name
        self.logdir = logdir

        if logdir:
            os.makedirs(logdir, exist_ok=True)

    def query(self, question: str) -> dict:
        result = self.pipeline.run(question)

        if self.logdir:
            self._log(result)

        return result.to_dict()

    def _log(self, result):
        path = os.path.join(self.logdir, f"{self.experiment_name}.jsonl")

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")