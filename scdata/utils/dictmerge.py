def dict_fmerge(base_dct, merge_dct, add_keys=True):
    """
    Recursive dict merge.
    From: https://gist.github.com/CMeza99/5eae3af0776bef32f945f34428669437
    Parameters
    ----------
        base_dct: dict
            Dict onto which the merge is executed
        merge_dct: dict
            Dict merged into base_dct
        add_keys: bool
            True
            Whether to add new keys
    Returns
    -------
        Updated dict
    """
    rtn_dct = base_dct.copy()
    if add_keys is False:
        merge_dct = {key: merge_dct[key] for key in set(rtn_dct).intersection(set(merge_dct))}

    rtn_dct.update({
        key: dict_fmerge(rtn_dct[key], merge_dct[key], add_keys=add_keys)
        if isinstance(rtn_dct.get(key), dict) and isinstance(merge_dct[key], dict)
        else merge_dct[key]
        for key in merge_dct.keys()
    })

    return rtn_dct
    