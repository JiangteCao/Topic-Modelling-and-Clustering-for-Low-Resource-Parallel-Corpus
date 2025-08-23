def clean_and_align(sorbian_sentences, german_sentences, max_len=128):
    """
    Clean and align Sorbian–German sentence pairs.

    Steps:
    1. Remove empty lines and strip whitespace.
    2. Truncate both sides to the same length.
    3. Filter out sentence pairs longer than max_len.
    4. Remove duplicate pairs.
    5. Return two aligned lists.
    """
    # Step 1: remove empty sentences
    sorbian_sentences = [s.strip() for s in sorbian_sentences if s.strip()]
    german_sentences = [s.strip() for s in german_sentences if s.strip()]

    # Step 2: keep same length
    min_len = min(len(sorbian_sentences), len(german_sentences))
    sorbian_sentences = sorbian_sentences[:min_len]
    german_sentences = german_sentences[:min_len]

    # Step 3: filter out too long sentences
    filtered_pairs = []
    for s, g in zip(sorbian_sentences, german_sentences):
        if len(s.split()) < max_len and len(g.split()) < max_len:
            filtered_pairs.append((s, g))

    # Step 4: remove duplicates
    seen = set()
    unique_pairs = []
    for pair in filtered_pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    # Step 5: unpack into two lists
    sorbian_clean, german_clean = zip(*unique_pairs) if unique_pairs else ([], [])
    return list(sorbian_clean), list(german_clean)
