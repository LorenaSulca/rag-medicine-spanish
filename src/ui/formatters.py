def format_chunk_title(chunk: dict) -> str:
    cid = chunk.get("chunk_id")
    rrf = chunk.get("rrf_rank")
    dk = chunk.get("dynamic_k")

    parts = [cid]

    if rrf:
        parts.append(f"rrf={rrf}")

    if dk:
        parts.append(f"k={dk}")

    return " | ".join(parts)


def format_validation_summary(validation: dict) -> str:
    usr = validation.get("usr")
    cit = validation.get("citation_consistency")

    multilevel = validation.get("multilevel") or {}
    numeric = multilevel.get("numeric_support_rate")

    return f"""
- **USR:** {usr}  
- **Citas:** {cit}  
- **Soporte numérico:** {numeric}  
- **Decisión:** {validation.get("decision")}
"""