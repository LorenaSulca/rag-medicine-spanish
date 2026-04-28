EXPERIMENTS = {
    "baseline": {
        "hybrid_retrieval": False,
        "citation_prompt": False,
        "sentence_validation": False,
        "dynamic_k": False,
        "refine_generation": False,
    },
    "p1_retrieval": {
        "hybrid_retrieval": True,
        "citation_prompt": False,
        "sentence_validation": False,
        "dynamic_k": False,
        "refine_generation": False,
    },
    "p1_citations": {
        "hybrid_retrieval": True,
        "citation_prompt": True,
        "sentence_validation": False,
        "dynamic_k": False,
        "refine_generation": False,
    },
    "propuesta_1_full": {
    "hybrid_retrieval": True,
    "citation_prompt": True,
    "sentence_validation": True,
    "dynamic_k": False,
    "refine_generation": False,
    "sentence_similarity_threshold": 0.20,
    "usr_partial_threshold": 0.01,
    "usr_invalid_threshold": 0.50,
    },
}