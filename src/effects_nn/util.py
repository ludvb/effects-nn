import warnings


def filter_state(state, *predicates):
    """
    Separate a nested state structure into multiple groups based on predicates.

    Args:
        state: Nested structure (dicts, tuples, NamedTuples, etc.)
        *predicates: Variable number of predicate functions to categorize leaf values

    Returns:
        (state_def, group1, group2, ..., groupN, rest) where:
        - state_def: Structure definition with placeholders
        - group1...groupN: Values matching each predicate
        - rest: Values not matching any predicate (always the last element)

    Example:
        >>> # Single predicate - returns matched and unmatched
        >>> state_def, params, rest = filter_state(state, lambda x: isinstance(x, jax.Array))

        >>> # Multiple predicates - returns groups plus rest
        >>> state_def, arrays, ints, rest = filter_state(state, is_array, is_int)
    """
    n_groups = len(predicates) + 1  # +1 for rest

    def traverse(obj, path=()):
        match obj:
            case None:
                return ("none", None), *([None] * n_groups)

            case dict():
                results = {k: traverse(v, path + (k,)) for k, v in obj.items()}
                state_defs = {k: r[0] for k, r in results.items()}
                groups = []
                for i in range(n_groups):
                    group = {k: r[i+1] for k, r in results.items() if r[i+1] is not None}
                    groups.append(group or None)
                return ("dict", state_defs), *groups

            case tuple() if hasattr(obj, "_fields"):  # NamedTuple
                results = {f: traverse(getattr(obj, f), path + (f,)) for f in obj._fields}
                state_defs = {f: r[0] for f, r in results.items()}
                groups = []
                for i in range(n_groups):
                    group = {f: r[i+1] for f, r in results.items() if r[i+1] is not None}
                    groups.append(group or None)
                return ("namedtuple", type(obj).__name__, type(obj), state_defs), *groups

            case tuple():
                results = [traverse(item, path + (i,)) for i, item in enumerate(obj)]
                state_defs = [r[0] for r in results]
                groups = []
                for i in range(n_groups):
                    group = tuple(r[i+1] for r in results)
                    # Only include if has non-None values
                    groups.append(group if any(v is not None for v in group) else None)
                return ("tuple", state_defs), *groups

            case list():
                results = [traverse(item, path + (i,)) for i, item in enumerate(obj)]
                state_defs = [r[0] for r in results]
                groups = []
                for i in range(n_groups):
                    group = [r[i+1] for r in results if r[i+1] is not None]
                    groups.append(group or None)
                return ("list", state_defs), *groups

            case _:  # Leaf value - test predicates
                # Test each predicate in order
                for i, pred in enumerate(predicates):
                    if pred(obj):
                        groups = [None] * n_groups
                        groups[i] = obj
                        return ("group", i, path), *groups

                # No predicate matched - goes to rest (last group)
                groups = [None] * n_groups
                groups[-1] = obj
                return ("rest", path), *groups

    return traverse(state)


def merge_state(state_def, *groups, strict: bool = False):
    """
    Reconstruct a state structure from state_def and groups.

    Args:
        state_def: Structure definition from filter_state
        *groups: group1, group2, ..., groupN, rest
                 (where rest contains unmatched values)
        strict: If False (default), ignore extra fields and warn about mismatches in NamedTuples.
                If True, raise errors on field mismatches (like PyTorch's load_state_dict).

    Returns:
        Reconstructed state matching the original structure

    Example:
        >>> state_def, arrays, rest = filter_state(state, is_array)
        >>> reconstructed = merge_state(state_def, arrays, rest)

        >>> state_def, arrays, ints, rest = filter_state(state, is_array, is_int)
        >>> reconstructed = merge_state(state_def, arrays, ints, rest)

        >>> # Gracefully handle structure mismatches
        >>> reconstructed = merge_state(state_def, arrays, rest, strict=False)
    """

    def reconstruct(def_node, group_nodes):
        if not isinstance(def_node, tuple):
            return None

        node_type = def_node[0]

        match node_type:
            case "none":
                return None

            case "dict":
                _, sub_defs = def_node
                return {
                    k: reconstruct(sub_def, [g.get(k) if g else None for g in group_nodes])
                    for k, sub_def in sub_defs.items()
                }

            case "namedtuple":
                _, name, cls, sub_defs = def_node

                # Reconstruct all values from sub_defs
                values = {
                    f: reconstruct(sub_def, [g.get(f) if g else None for g in group_nodes])
                    for f, sub_def in sub_defs.items()
                }

                # Get expected fields from the NamedTuple class
                expected_fields = set(cls._fields)
                provided_fields = set(values.keys())

                # Check for mismatches
                extra_fields = provided_fields - expected_fields
                missing_fields = expected_fields - provided_fields

                if extra_fields or missing_fields:
                    if strict:
                        # In strict mode, raise an error
                        error_parts = []
                        if extra_fields:
                            error_parts.append(f"extra fields {extra_fields}")
                        if missing_fields:
                            error_parts.append(f"missing fields {missing_fields}")
                        raise ValueError(
                            f"NamedTuple {name} field mismatch: {', '.join(error_parts)}"
                        )
                    else:
                        # In non-strict mode, warn and filter
                        if extra_fields:
                            warnings.warn(
                                f"Ignoring extra fields {extra_fields} when reconstructing {name}",
                                UserWarning,
                                stacklevel=3
                            )
                        if missing_fields:
                            warnings.warn(
                                f"Missing fields {missing_fields} when reconstructing {name}, "
                                f"will use default values if available",
                                UserWarning,
                                stacklevel=3
                            )

                # Get default values for the NamedTuple if they exist
                defaults = {}
                if hasattr(cls, '_field_defaults'):
                    # typing.NamedTuple style
                    defaults = cls._field_defaults
                elif hasattr(cls, '__defaults__') and cls.__defaults__:
                    # collections.namedtuple style - defaults are for last N fields
                    n_defaults = len(cls.__defaults__)
                    for i, default in enumerate(cls.__defaults__):
                        field_idx = len(cls._fields) - n_defaults + i
                        defaults[cls._fields[field_idx]] = default

                # Filter to only expected fields, using defaults when available
                filtered_values = {
                    f: values.get(f, defaults.get(f, None)) for f in expected_fields
                }

                return cls(**filtered_values)

            case "tuple":
                _, sub_defs = def_node
                # Need to track position in each group's tuple
                result = []
                indices = [0] * len(group_nodes)

                for sub_def in sub_defs:
                    # Get values from appropriate positions in tuple groups
                    sub_groups = []
                    for i, group in enumerate(group_nodes):
                        if group and isinstance(group, tuple) and indices[i] < len(group):
                            value = group[indices[i]]
                            # Only advance if this position contributes to this group
                            if value is not None or sub_def[0] in ("dict", "namedtuple", "tuple", "list"):
                                indices[i] += 1
                            sub_groups.append(value)
                        else:
                            sub_groups.append(None)

                    result.append(reconstruct(sub_def, sub_groups))

                return tuple(result)

            case "list":
                _, sub_defs = def_node
                return [
                    reconstruct(sub_def, [g[i] if g and i < len(g) else None for g in group_nodes])
                    for i, sub_def in enumerate(sub_defs)
                ]

            case "group":
                _, group_idx, _ = def_node
                return group_nodes[group_idx]

            case "rest":
                _, _ = def_node
                return group_nodes[-1]  # Rest is always last

            case _:
                raise ValueError(f"Unknown node type: {node_type}")

    return reconstruct(state_def, list(groups))
