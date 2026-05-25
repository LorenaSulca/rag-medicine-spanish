BASE_THRESHOLDS = {
    "sentence_similarity_threshold": 0.20,
    "usr_partial_threshold": 0.01,
    "usr_invalid_threshold": 0.50,
}


EXPERIMENTS = {
    # =========================
    # Experimentos originales
    # =========================

    "baseline": {
        "index_variant": "sections",
        "hybrid_retrieval": False,
        "citation_prompt": False,
        "sentence_validation": False,
        "dynamic_k": False,
        "refine_generation": False,
    },

    "p1_retrieval": {
        "index_variant": "sections",
        "hybrid_retrieval": True,
        "citation_prompt": False,
        "sentence_validation": False,
        "dynamic_k": False,
        "refine_generation": False,
    },

    "p1_citations": {
        "index_variant": "sections",
        "hybrid_retrieval": True,
        "citation_prompt": True,
        "sentence_validation": False,
        "dynamic_k": False,
        "refine_generation": False,
    },

    "propuesta_1_full": {
        "index_variant": "sections",
        "hybrid_retrieval": True,
        "citation_prompt": True,
        "sentence_validation": True,
        "dynamic_k": False,
        "refine_generation": False,
        **BASE_THRESHOLDS,
    },

    "p2_dynamic_retrieval": {
        "index_variant": "sections",
        "hybrid_retrieval": True,
        "dynamic_k": True,
        "citation_prompt": True,
        "sentence_validation": True,
        "refine_generation": False,
        "multi_validation": False,
        **BASE_THRESHOLDS,
    },

    "p2_refine": {
        "index_variant": "sections",
        "hybrid_retrieval": True,
        "dynamic_k": True,
        "citation_prompt": True,
        "sentence_validation": True,
        "refine_generation": True,
        "multi_validation": False,
        **BASE_THRESHOLDS,
    },

    "propuesta_2_full": {
        "index_variant": "sections",
        "hybrid_retrieval": True,
        "dynamic_k": True,
        "citation_prompt": True,
        "sentence_validation": True,
        "refine_generation": True,
        "multi_validation": True,
        **BASE_THRESHOLDS,
    },

    # =========================
    # Experimentos: flat vs sections
    # =========================

    "baseline_flat": {
        "index_variant": "flat",
        "hybrid_retrieval": False,
        "citation_prompt": False,
        "sentence_validation": False,
        "dynamic_k": False,
        "refine_generation": False,
    },

    "baseline_sections": {
        "index_variant": "sections",
        "hybrid_retrieval": False,
        "citation_prompt": False,
        "sentence_validation": False,
        "dynamic_k": False,
        "refine_generation": False,
    },

    "propuesta_1_full_flat": {
        "index_variant": "flat",
        "hybrid_retrieval": True,
        "citation_prompt": True,
        "sentence_validation": True,
        "dynamic_k": False,
        "refine_generation": False,
        **BASE_THRESHOLDS,
    },

    "propuesta_1_full_sections": {
        "index_variant": "sections",
        "hybrid_retrieval": True,
        "citation_prompt": True,
        "sentence_validation": True,
        "dynamic_k": False,
        "refine_generation": False,
        **BASE_THRESHOLDS,
    },

    "propuesta_2_full_flat": {
        "index_variant": "flat",
        "hybrid_retrieval": True,
        "dynamic_k": True,
        "citation_prompt": True,
        "sentence_validation": True,
        "refine_generation": True,
        "multi_validation": True,
        **BASE_THRESHOLDS,
    },

    "propuesta_2_full_sections": {
        "index_variant": "sections",
        "hybrid_retrieval": True,
        "dynamic_k": True,
        "citation_prompt": True,
        "sentence_validation": True,
        "refine_generation": True,
        "multi_validation": True,
        **BASE_THRESHOLDS,
    },
}