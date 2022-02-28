
import streamlit as st
import streamlit.components.v1 as components
import streamlit
import pandas as pd
import numpy as np
import networkx as nx
import wikipediaapi
from pyvis.network import Network

book1 = pd.read_csv("got_dataset/book1.csv")
book2 = pd.read_csv("got_dataset/book2.csv")
book3 = pd.read_csv("got_dataset/book3.csv")
book4 = pd.read_csv("got_dataset/book4.csv")
book5 = pd.read_csv("got_dataset/book5.csv")


def gen_network(df):
    net = Network(height='750px', width='100%',
                  bgcolor='#0E1117', font_color='white')
    net.barnes_hut()
    sources = df['Source']
    targets = df['Target']
    weights = df['weight']
    g_book = nx.Graph()
    for index, edge in df.iterrows():
        g_book.add_edge(edge['Source'], edge['Target'], weight=edge['weight'])
    edge_data = zip(sources, targets, weights)
    degree = nx.degree(g_book)

    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]

        net.add_node(src, src, title=src)
        net.add_node(dst, dst, title=dst)
        net.add_edge(src, dst, value=w)

    deg_cen = nx.degree_centrality(g_book)
    bet_cen = nx.betweenness_centrality(g_book, weight='weight')
    close_cen = nx.closeness_centrality(g_book)
    eigen_cen = nx.eigenvector_centrality(g_book, weight='weight')
    #page_rank_cen = nx.pagerank(g_book, weight='weight')

    n_nodes = g_book.number_of_nodes()
    n_edges = g_book.number_of_edges()
    avg_short_path = np.round(nx.average_shortest_path_length(g_book), 2)
    diameter = nx.algorithms.distance_measures.diameter(g_book)

    for node in net.nodes:
        node['title'] += '<br>Node Degree: '+str(degree[node['id']]) + ' <br>Betweenness Centrality: ' + '{0:.4f}'.format(bet_cen[node['id']]) + '<br>Degree Centrality: ' + '{0:.4f}'.format(
            deg_cen[node['id']])+'<br>Closeness Centrality: ' + '{0:.4f}'.format(close_cen[node['id']]) + '<br>Eigenvector Centrality: ' + '{0:.4f}'.format(eigen_cen[node['id']])
        node['value'] = 1000*degree[node['id']]
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidthSelected": 4,
        "color": {
          "highlight": {
            "border": "rgba(0,255,255,1)",
            "background": "rgba(0,191,255,1)"
          },
          "hover": {
            "border": "rgba(0,139,139,1)",
            "background": "rgba(0,255,255,1)"
          }
        },
        "font": {
          "background": "rgba(0,0,0,0)"
        },
        "shadow": {
          "enabled": true
        },
        "shapeProperties": {
          "borderRadius": 4
        }
      },
      "edges": {
        "color": {
          "highlight": "rgba(59,127,132,1)",
          "inherit": false,
          "opacity": 0.45
        },
        "smooth": {
          "type": "continuous",
          "forceDirection": "none",
          "roundness": 0.2
        }
      },
      "interaction": {
        "hover": true,
        "multiselect": true
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "springLength": 325,
          "springConstant": 0.03,
          "damping": 0.41,
          "avoidOverlap": 0.45
        },
        "minVelocity": 0.75
      }
    }
    """)
    return (net, n_nodes, n_edges, avg_short_path, diameter)


# Set header title
st.set_page_config(layout="wide")
st.title('Game of Thrones Network Analysis Dashboard')
#st.markdown("This is a dashboard for analysis the GOT character network")
st.image("got_dataset/wallpaper.jpg")

base = st.get_option("theme.base")
primary_color = st.get_option("theme.primaryColor")
base = "dark"
primaryColor = "#35b6dc"

col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
with col2:
    book_list = ['A Game of Thrones', 'A Clash of Kings',
                 'A Storm of Swords', 'A Feast for Crows', 'A Dance with Dragons']
    book = st.selectbox("Select Book:", book_list)
    # # Define list of selection options and sort alphabetically
    # char_list = ['Eddard-Stark', 'Bran-Stark', 'Arya-Stark',
    #              'Daenerys-Targaryen', 'Jon-Snow', 'Cersei-Lannister']
    # char_list.sort()
    #
    # # Implement multiselect dropdown menu for option selection (returns a list)
    # selected_chars = st.multiselect('Select character(s) to visualize', char_list)

    # Set info message on initial site load
    if book == book_list[0]:
        df_select = book1
    elif book == book_list[1]:
        df_select = book2
    elif book == book_list[2]:
        df_select = book3
    elif book == book_list[3]:
        df_select = book4
    else:
        df_select = book5

    got_net, n_nodes, n_edges, avg_short_path, diameter = gen_network(
        df_select)
    # Save and read graph as HTML file (on Streamlit Sharing)
    try:
        got_net.save_graph('pyvis_graph.html')
        HtmlFile = open('pyvis_graph.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
    except:
        got_net.save_graph('pyvis_graph.html')
        HtmlFile = open('pyvis_graph.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=1000)

#  Network Properties
with col1:
    st.subheader("Network Properties")
    st.write("Number of nodes:", str(n_nodes))
    st.write("Number of edges:", str(n_edges))
    st.write("Diameter of Network:", str(diameter))
    st.write("Average Shortest Path:", str(avg_short_path))
    properties = ['Node Degree', 'Betweenness Centrality', 'Degree Centrality',
                  'Closeness Centrality', 'Eigenvector Centrality']
    st.subheader("Centrality Measures")
    property = st.multiselect('Select Measure:', properties)
    if 'Node Degree' in property:
        st.subheader('Node Degree')
        st.write(
            'The degree of a node is the number of edges connected to the node. In other words, it is the number of neighbours a node has. In terms of the adjacency matrix A, \
            the degree for a node indexed by i in an undirected network is given by:')
        st.latex(r'''k_i=\sum_{j}a_{ij}
        ''')
        st.write('where the sum is over all the nodes in the network.',
                 'Node degree is an effective measure of the influence or importance of a node.')
    if 'Betweenness Centrality' in property:
        st.subheader('Betweenness Centrality')

        st.write('Betweenness Centrality is a measure of importance of a node in terms of the connection it creates among other nodes. For example, a node can have a small degree centrality but it might play an important role in keeping together clusters of several nodes. It quantifies the number of times a node acts as a bridge along the shortest path between two other nodes.')
        st.write(
            'The betweenness centrality of a node i in a graph G is computed as follows:')
        st.write(
            '1. For each pair of vertices, compute the shortest path between them.')
        st.write('2. For each pair of vertices, determine the fraction of shortest paths that pass through the given vertex.')
        st.write('3. Sum this fraction over all pairs of vertices.')
        st.latex(r''' b_i=\sum_{h\neq j \neq i}\frac{\sigma_{hj}(i)}{\sigma_{hj}}
        ''')
        st.write('where the denominator represents the total number of shortest paths between h and j and the numerator represents the number of those paths that pass through node i.')
    if 'Degree Centrality' in property:
        st.subheader('Degree Centrality')
        st.write(
                     'Degree Centrality measures the number of edges incident on a node.')
    if 'Closeness Centrality' in property:
        st.subheader('Closeness Centrality')
        st.write(
            'Closeness centrality measures importance of a node by how close it is to all other nodes in the graph.')
        st.write('Let dij be the length of the shortest path between nodes i andd j. The average distance of node i is given by')
        st.latex(r'''
        l_{i}=\frac{1}{n}\sum_{j}d_{ij}
        ''')
        st.write(
            'The closeness centrality is inversely proportional to the average length, or is a reciprocal of farness, thus')
        st.latex(r'''
        C_i=\frac{1}{l_i}=\frac{n}{\sum_{j}d_{ij}}''')
        st.write('Since we have an undirected graph, all edges have distance 1.')

        #st.write('Closeness Centrality is sum of')
    if 'Eigenvector Centrality' in property:
        st.subheader('Eigenvector Centrality')
        st.write('Eigenvector Centrality defines centrality of a node as proportional to its neighbours\' importance. It is based on the idea that connections to high scoring nodes contribute more to the importance of a node as compared to equal collections to low scoring nodes. ')
        st.write('For a graph G, the vertices V and edges E, an entry in the adjacency matrix A has a value 1 if a vertex i is linked to a vertex j and a value 0 otherwise. The relative eigenvector centrality score of vertex i can be defined as,')
        st.latex(
            r'''X_{i}=\frac{1}{\lambda}\sum_{j \in M(i)}{X_j}=\frac{1}{\lambda}\sum_{j \in G}a_{i,j}X_{j}''')
        st.write(
            "where M(i) is the set of all neighbours of i and lambda is a constant.")
        st.write("This equation is defined recursively and it requires finding the eigenvector centrality of all neighbour nodes. The original equation can be written in vector notation as:")
        st.latex(r'''
        Ax=\lambda x''')
        st.write("This equation can be solved using linear algebra to find the value of lambda. The greatest eigenvector gives us the centrality scores (by Perron-Frobenius theorem). Here the condition is that A is a positive matrix, which is true since it is an adjacency matrix. ")
        # st.write("What multiplication by the adjacency matrix does, is reassign each vertex the sum of the values of its neighbor vertices.This has, in effect, spreads out the degree centrality. Suppose we multiplied the resulting vector by A again, in effect, we'd be allowing this centrality value to once again spread across the edges of the graph.")
    # if 'Page Rank Centrality' in property:
    #     st.subheader('Page Rank Centrality')
    #     st.write('Page Rank Centrality is a variation of Eigenvector Centrality. It is mostly applicable for directed networks. The idea here is that nodes with many incoming links are influencial and the nodes to which they are connected share some of that influence.')

    with col3:
        char_list = list(df_select.Source.unique())
        selected_char = st.selectbox("Select Character", char_list)
        st.subheader(selected_char.replace('-', ' '))
        wiki_wiki = wikipediaapi.Wikipedia('en')
        selected_char = selected_char.replace('-', '_')
        page = wiki_wiki.page(selected_char)
        #print(page.summary)
        st.write(page.summary)

#https://u.osu.edu/nix.39/2021/01/22/breaking-down-the-eigenvector-centrality-measure/
