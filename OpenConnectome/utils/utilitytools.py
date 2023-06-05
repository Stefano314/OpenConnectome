import networkx as nx

def getNetworkComponents(G : nx.Graph) -> list:
    """
    Description
    -----------
    Build a list with all the components in a generic network and the nodes appearing in each component.

    """

    components = []
    components.append(list(nx.node_connected_component(G, list(G.nodes())[0])))

    for n in list(G.nodes())[1:]:
        
        found = False
        for component in components:
            if n in component:
                found = True

        if not found:
            components.append(list(nx.node_connected_component(G,n)))

    return components


def drawComponentsSizes(graph_components : list):
    """
    Description
    -----------
    Simple plot of the components sizes in a generic network. This function is meant to be paired with getNetworkComponents().
    """

    import matplotlib.pyplot as plt

    # Extract the lengths of sublists from each list
    lengths = [[len(subsublist) for subsublist in sublist] for sublist in graph_components]

    for component,sublist in enumerate(lengths):

        plt.bar([f'G{component}_{i}' for i in range(len(sublist))], sublist, width=1, label=f'C{component}')


    # Show the plot
    plt.xticks([])
    plt.yscale('log')
    plt.title("Connected Components Lengths", fontsize=15, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()