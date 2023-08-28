class Kcal2meV:
    def __init__(self):
        # Kcal/mol to meV
        self.conversion = 43.3634

    def __call__(self, graph):
        graph.energy = graph.energy * self.conversion
        graph.force = graph.force * self.conversion
        return graph
