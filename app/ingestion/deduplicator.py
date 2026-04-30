import hashlib

def is_duplicate(text:str, seen_hashes:set)->bool:
    hashed_str = hashlib.md5(text.lower().strip().encode('utf-8')).hexdigest()

    if hashed_str in seen_hashes:
        return True
    seen_hashes.add(hashed_str)

    return False