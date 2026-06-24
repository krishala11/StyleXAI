"""Online recommendation pipeline orchestrator."""
from src.logging_utils import append_trace, save_checkpoint
from src.online.explanation import generate_explanation
from src.online.filter import filter_candidates
from src.online.intent import extract_intent
from src.online.outfit_builder import construct_outfits
from src.online.ranking import rank_outfits
from src.online.retrieval import score_retrieval


def recommend(kb, profile: dict, query: str, mode: int = 1, selections: dict | None = None) -> dict:
    """
    mode=1: full chat recommendation (top 3 outfits)
    mode=2: guided builder with stepwise selections
    """
    print(f"\n[PIPELINE] Received query: {query}")
    trace = []
    append_trace("user_input", {"profile": profile, "query": query, "mode": mode}, trace)

    print("[PIPELINE] 1. Extracting Intent via LLM/Rules...")
    intent = extract_intent(query, profile, trace)
    
    source = intent.get("source", "unknown").upper()
    print(f"[PIPELINE] Intent extracted via: {source}")
    print(f"[PIPELINE] Parsed Intent: Occasion={intent.get('occasion')} | Style={intent.get('style')}")

    print("[PIPELINE] 2. Filtering Candidates...")
    pools = filter_candidates(kb, intent, trace)
    print(f"[PIPELINE] Filtering done. Pool sizes: {{k: len(v) for k, v in pools.items()}}")

    print("[PIPELINE] 3. FAISS Retrieval & Scoring...")
    ranked = score_retrieval(kb, pools, intent, query, trace)
    print(f"[PIPELINE] Retrieval done.")
    
    save_checkpoint("checkpoint_4_retrieval", {
        "intent": intent,
        "pool_counts": {k: len(v) for k, v in pools.items()},
        "top_retrieval": {k: v[:5] for k, v in ranked.items()},
    })

    if mode == 2 and selections:
        return _guided_builder(kb, profile, query, intent, pools, ranked, selections, trace)

    paths = construct_outfits(kb, pools, ranked, intent, trace)
    top_outfits = rank_outfits(kb, paths, ranked, profile, intent, trace)

    print("[PIPELINE] 5. Generating Explanations...")
    for outfit in top_outfits:
        outfit["explanation"] = generate_explanation(outfit, intent, profile, trace)
    
    # Just grab the last explanation event from the trace to see what was used
    if trace and "explanation_" in trace[-1]["event"]:
        print(f"[PIPELINE] Explanations generated via: {trace[-1]['event'].replace('explanation_', '').upper()}")

    result = {
        "mode": mode,
        "intent": intent,
        "outfits": top_outfits,
        "trace": trace,
    }
    save_checkpoint("checkpoint_6_final_output", result)
    return result


def _guided_builder(kb, profile, query, intent, pools, ranked, selections, trace):
    """Mode 2 stepwise: selections = {topwear?, bottomwear?, footwear?, accessory?}"""
    from src.online.compatibility import pair_compatibility
    from src.online.expansion import expand_neighbors
    from src.online.ranking import outfit_level_scores
    from src.online.explanation import generate_explanation

    def _gender_pool(pool_ids: list[str]) -> list[str]:
        gender = intent.get("gender", profile.get("gender", "men"))
        return [
            pid for pid in pool_ids
            if str(kb.products[pid].get("gender", "")).lower() in (gender, "unisex", "")
        ]

    def _score_slot_candidates(anchor: str, slot: str, pool_ids: list[str]) -> list[dict]:
        """Rank candidates by existing pair_compatibility + retrieval (guided UI only)."""
        if not anchor:
            return []
        pool_ids = _gender_pool(pool_ids)
        if not pool_ids:
            pool_ids = _gender_pool([
                pid for pid, _ in kb.products.items() if kb.get_slot(pid) == slot
            ])

        # Graph neighbors first — uses existing graph expansion logic
        neighbor_ids = [n["id"] for n in expand_neighbors(kb, anchor, trace)]
        ordered_ids = []
        seen = set()
        for pid in neighbor_ids + pool_ids:
            if pid and pid not in seen and pid != anchor and kb.get_slot(pid) == slot:
                seen.add(pid)
                ordered_ids.append(pid)

        from src.online.filter import _occasion_match, _style_match
        scored = []
        for pid in ordered_ids:
            pc = pair_compatibility(kb, anchor, pid, intent)
            if not pc["rejected"]:
                product = kb.products[pid]
                occ_match = _occasion_match(product, intent.get("occasion", ""))
                style_match = _style_match(product, intent.get("style", ""), intent)
                is_strict = occ_match and style_match
                strict_bonus = 1.0 if is_strict else 0.0

                ret = next((r["retrieval_score"] for r in ranked.get(slot, []) if r["id"] == pid), 0)
                graph_bonus = 0.12 if pid in neighbor_ids else 0.0
                scored.append({
                    "id": pid,
                    "name": kb.products[pid]["name"],
                    "image": kb.products[pid]["image"],
                    "pair_score": pc["pair_score"],
                    "retrieval_score": ret,
                    "sort_score": pc["pair_score"] + 0.1 * ret + graph_bonus + strict_bonus,
                    "from_graph": pid in neighbor_ids,
                    "is_strict_match": is_strict
                })

        if not scored:
            for pid in ordered_ids[:12]:
                product = kb.products[pid]
                occ_match = _occasion_match(product, intent.get("occasion", ""))
                style_match = _style_match(product, intent.get("style", ""), intent)
                is_strict = occ_match and style_match
                strict_bonus = 1.0 if is_strict else 0.0

                ret = next((r["retrieval_score"] for r in ranked.get(slot, []) if r["id"] == pid), 0)
                scored.append({
                    "id": pid,
                    "name": kb.products[pid]["name"],
                    "image": kb.products[pid]["image"],
                    "pair_score": 0.49,
                    "retrieval_score": ret,
                    "sort_score": 0.49 + 0.1 * ret + strict_bonus,
                    "from_graph": pid in neighbor_ids,
                    "is_strict_match": is_strict
                })

        scored.sort(key=lambda x: -x["sort_score"])
        return scored[:12]

    step = selections.get("step", 1)
    response = {"mode": 2, "step": step, "intent": intent, "trace": trace}

    if step == 1:
        # Guarantee Mode 1 tops are shown first
        from src.online.outfit_builder import construct_outfits
        from src.online.ranking import rank_outfits
        
        # Mirror mode 1 exact ranking to get the real top outfits
        paths = construct_outfits(kb, pools, ranked, intent, trace)
        top_outfits = rank_outfits(kb, paths, ranked, profile, intent, trace)
        
        mode1_tops = []
        for outfit in top_outfits:
            tid = outfit["items"].get("topwear", {}).get("id")
            if tid and tid not in mode1_tops:
                mode1_tops.append(tid)
        
        seen_ids = set()
        candidates = []
        
        def add_candidate(pid):
            if pid in seen_ids or not pid: return
            seen_ids.add(pid)
            from src.online.filter import _occasion_match, _style_match
            product = kb.products[pid]
            occ_match = _occasion_match(product, intent.get("occasion", ""))
            style_match = _style_match(product, intent.get("style", ""), intent)
            is_strict = occ_match and style_match
            
            ret = next((r["retrieval_score"] for r in ranked.get("topwear", []) if r["id"] == pid), 0)
            candidates.append({
                "id": pid,
                "name": kb.products[pid]["name"],
                "image": kb.products[pid]["image"],
                "retrieval_score": ret,
                "is_strict_match": is_strict
            })

        for tid in mode1_tops:
            add_candidate(tid)
            
        for r in ranked.get("topwear", []):
            add_candidate(r["id"])
            if len(candidates) >= 12:
                break
                
        # Sort so strict matches bubble to the top
        for c in candidates:
            c["sort_score"] = (1.0 if c.get("is_strict_match") else 0.0) + (0.1 * c.get("retrieval_score", 0))
        candidates.sort(key=lambda x: -x["sort_score"])
                
        response["candidates"] = candidates[:12]
        response["message"] = "Select a topwear item"
        return response

    if step == 2:
        top_id = selections.get("topwear")
        bottom_pool = pools.get("bottomwear", [])
        if not bottom_pool:
            bottom_pool = [pid for pid in kb.products if kb.get_slot(pid) == "bottomwear"]

        if kb.is_one_piece(top_id):
            response["skip_bottom"] = True
            response["candidates"] = []
            response["message"] = "One-piece outfit — continue to footwear"
        else:
            response["candidates"] = _score_slot_candidates(top_id, "bottomwear", bottom_pool)
            response["message"] = "Select bottomwear"
        return response

    if step == 3:
        anchor = selections.get("bottomwear") or selections.get("topwear")
        footwear_pool = pools.get("footwear", [])
        if not footwear_pool:
            footwear_pool = [pid for pid in kb.products if kb.get_slot(pid) == "footwear"]
        response["candidates"] = _score_slot_candidates(anchor, "footwear", footwear_pool)
        response["message"] = "Select footwear"
        return response

    if step == 4:
        anchor = selections.get("topwear")
        accessory_pool = pools.get("accessory", [])
        if not accessory_pool:
            accessory_pool = [pid for pid in kb.products if kb.get_slot(pid) == "accessory"]
        response["candidates"] = _score_slot_candidates(anchor, "accessory", accessory_pool)
        response["message"] = "Select an accessory (optional — you can skip)"
        return response

    # Step 5: final outfit
    path = {
        "topwear": selections.get("topwear"),
        "bottomwear": selections.get("bottomwear"),
        "footwear": selections.get("footwear"),
        "accessory": selections.get("accessory"),
        "pairs": [],
    }
    ids = [path["topwear"], path["bottomwear"], path["footwear"], path["accessory"]]
    ids = [i for i in ids if i]
    if path["topwear"] and path["bottomwear"]:
        path["pairs"].append(pair_compatibility(kb, path["topwear"], path["bottomwear"], intent))
    anchor = path["bottomwear"] or path["topwear"]
    if anchor and path["footwear"]:
        path["pairs"].append(pair_compatibility(kb, anchor, path["footwear"], intent))
    if path["topwear"] and path["accessory"]:
        path["pairs"].append(pair_compatibility(kb, path["topwear"], path["accessory"], intent))

    scores = outfit_level_scores(kb, path, ranked, profile, intent)
    items = {}
    for slot in ("topwear", "bottomwear", "footwear", "accessory"):
        pid = path.get(slot)
        if pid:
            items[slot] = {
                "id": pid,
                "name": kb.products[pid]["name"],
                "image": kb.products[pid]["image"],
                "brand": kb.products[pid].get("brand"),
            }
    outfit = {"items": items, "scores": scores}
    outfit["explanation"] = generate_explanation(outfit, intent, profile, trace)
    response["outfits"] = [outfit]
    response["live_compatibility"] = scores["compatibility_score"]
    save_checkpoint("checkpoint_6_final_output", response)
    return response


def get_compatible_options(kb, profile: dict, query: str, slot: str, selected: dict) -> list:
    """Helper for guided mode live preview."""
    trace = []
    intent = extract_intent(query, profile, trace)
    pools = filter_candidates(kb, intent, trace)
    ranked = score_retrieval(kb, pools, intent, query, trace)

    if slot == "topwear":
        return ranked.get("topwear", [])[:12]
    if slot == "bottomwear" and selected.get("topwear"):
        from src.online.compatibility import pair_compatibility
        results = []
        for pid in pools.get("bottomwear", []):
            pc = pair_compatibility(kb, selected["topwear"], pid, intent)
            if not pc["rejected"]:
                results.append({**kb.products[pid], "pair_score": pc["pair_score"]})
        results.sort(key=lambda x: -x["pair_score"])
        return results[:12]
    return []
