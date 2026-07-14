import json
import collections


def main():
    with open(
        "../data/qa/main_qa_5_prospectos_merged.json",
        encoding="utf-8"
    ) as f:
        d = json.load(f)

    for field in [
        "document_id",
        "category",
        "difficulty",
        "expected_behavior",
    ]:
        print(f"\n{field}")

        counter = collections.Counter(x[field] for x in d)
        for k, v in counter.items():
            print(k, v)

    print("\nnumeric", sum(x["requires_numeric_grounding"] for x in d))
    print("multihop", sum(x["requires_multi_hop"] for x in d))


if __name__ == "__main__":
    main()