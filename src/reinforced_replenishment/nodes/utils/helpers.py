def get_values_by_key(nested_dict, target_key):
    results = []

    def search_dict(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if key == target_key:
                    results.append(value)
                search_dict(value)
        elif isinstance(d, list):
            for item in d:
                search_dict(item)

    search_dict(nested_dict)
    return results


def delete_keys_from_nested_dict(d: dict, keys_to_delete: list[str]):
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if key not in keys_to_delete:
                new_dict[key] = delete_keys_from_nested_dict(value, keys_to_delete)
        return new_dict
    elif isinstance(d, list):
        return [delete_keys_from_nested_dict(item, keys_to_delete) for item in d]
    else:
        return d
