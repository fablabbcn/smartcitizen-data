def topological_sort(items):
    '''
    Returns a topological sort for items in a list that depend on each other
    Each item has:
    class Item:
        name: str
        depends_on: [str]
    Returns:
        Sorted list of items
    '''
    from collections import defaultdict

    # Create a graph and an indegree count
    graph = defaultdict(list)
    indegree = {item.name: 0 for item in items}

    # Build the graph
    for item in items:
        for dependency in item.depends_on:
            graph[dependency].append(item.name)
            indegree[item.name] += 1

    # Collect nodes with no incoming edges (indegree of 0)
    queue = [item for item in items if indegree[item.name] == 0]
    ordered_items = []

    while queue:
        current = queue.pop(0)
        ordered_items.append(current)

        # Decrease the indegree of neighbors
        for neighbor in graph[current.name]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(next(item for item in items if item.name == neighbor))

    # Check if there are any remaining items with indegrees
    if len(ordered_items) != len(items):
        return None

    return ordered_items