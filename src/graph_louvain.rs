#[cfg(test)]
#[path = "./graph_test.rs"]
mod graph_test;

use core::panic;
use std::collections::{HashMap, HashSet};

pub type Node = usize;
pub type EdgeWeight = f64;

/// "Adj" stands for "adjacent".
/// Edges are stored in a HashMap where the key is the node id and
/// the value is a vector of tuples of the form (neighbor, weight).
pub type Adj = HashMap<Node, HashMap<Node, EdgeWeight>>;

/// An undirected graph of vertices and edges with weights.
///
/// Mathematically, a graph is a tuple G = (V, E) of vertices V and edges E.
/// We use an adjacency list representation of our graph, i.e.,
/// a map from nodes to their respective neighbors (including weights)
/// as this information is retrieved frequently in the Louvain algorithm.
///
/// Nodes are contiguously labeled from 0 to n-1 (at most). Note that the graph
/// does not enforce the capactiy to be fully used. There might be isolated
/// nodes in the graph.
#[derive(Debug)]
pub struct Graph {
    pub adj: Adj,
}

impl Graph {
    pub fn new(initial_capacity: usize) -> Self {
        Graph {
            adj: HashMap::with_capacity(initial_capacity),
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.adj.len()
    }

    pub fn insert_edge(&mut self, source: Node, target: Node, weight: EdgeWeight) {
        if !self.assert_edge_does_not_exist(source, target) {
            return;
        }

        // Graph is undirected, so we add the edge in both directions
        let neighbors = self.adj.entry(source).or_insert(HashMap::new());
        neighbors.insert(target, weight);

        let is_self_loop = self.is_self_loop(source, target);
        if !is_self_loop {
            let neighbors = self.adj.entry(target).or_insert(HashMap::new());
            neighbors.insert(source, weight);
        }
    }

    /// Returns an iterator over all edges of the graph.
    ///
    /// Makes sure that each edge is only visited once. Only edges
    /// with source <= target are visited. The order in which edges
    /// are visited is not specified.
    pub fn edges(&self) -> impl Iterator<Item = (Node, Node, EdgeWeight)> + '_ {
        self.adj.iter().flat_map(|(source, neighbors)| {
            neighbors
                .iter()
                // avoid visiting the same edge twice
                .filter(move |(target, _)| source <= target)
                .map(move |(target, weight)| (*source, *target, *weight))
        })
    }

    /// Increases the weight of an edge.
    ///
    /// Panics if the edge does not exist in the graph yet.
    pub fn increase_edge_weight(&mut self, source: Node, target: Node, weight_delta: EdgeWeight) {
        self.assert_edge_exists(source, target);

        self.adj.entry(source).and_modify(|neighbors| {
            neighbors.entry(target).and_modify(|w| {
                *w += weight_delta;
            });
        });

        if source != target {
            self.adj.entry(target).and_modify(|neighbors| {
                neighbors.entry(source).and_modify(|w| {
                    *w += weight_delta;
                });
            });
        }
    }

    /// Returns the adjacent edges of a node.
    ///
    /// This might include self-loops.
    pub fn adjacent_edges(&self, node: Node) -> Option<&HashMap<Node, EdgeWeight>> {
        return self.adj.get(&node);
    }

    /// Returns the adjacent nodes of a node.
    ///
    /// This does not include the node itself if it has a self-loop.
    pub fn adjacent_nodes(&self, node: Node) -> Option<HashSet<Node>> {
        let neighbors = self.adjacent_edges(node);

        let res: HashSet<usize> = neighbors?
            .keys()
            .filter(|&other| !self.is_self_loop(node, *other))
            .copied() // no worries, we use usize for "Node"
            .collect();
        Some(res)
    }

    /// "Finalizes the graph" by making sure that nodes are contiguously labeled.
    ///
    /// If this is not the case, it panics, as we need a contiguous labeling
    /// for the Louvain algorithm later on.
    pub fn finalize(&mut self) {
        self.assert_nodes_contiguously_labeled();
    }

    /// Checks if nodes are contiguously labeled, i.e. [0,1,2,3,...] instead of
    /// [0,3,4,5,6,...] (missing out the 2 in the example).
    ///
    /// Panics if nodes are not contiguously labeled.
    fn assert_nodes_contiguously_labeled(&self) {
        let mut nodes = self.adj.keys().collect::<Vec<_>>();
        nodes.sort(); // TODO: this is costly! Maybe add an option to disable this check?
                      // if user knows for sure that nodes are contiguously labeled.

        for (expected_node_id, node) in nodes.into_iter().enumerate() {
            if *node != expected_node_id {
                panic!(
                    "Nodes are not contiguously labeled. Please make sure you're \
                    graph does not contain isolated nodes. Currently missing node \
                    {}.",
                    expected_node_id
                );
            }
        }
    }

    fn is_self_loop(&self, source: Node, target: Node) -> bool {
        source == target
    }

    fn assert_node_exists(&self, node: &Node) {
        if self.adj.contains_key(node) {
            return;
        }
        panic!("Node {} does not exist in graph", node);
    }

    fn assert_edge_exists(&self, source: Node, target: Node) {
        self.assert_node_exists(&source);
        self.assert_node_exists(&target);

        if !self.adj[&source].contains_key(&target) {
            panic!("Edge ({} <-> {}) does not exist in graph", source, target);
        }
    }

    fn assert_edge_does_not_exist(&self, source: Node, target: Node) -> bool {
        if !self.adj.contains_key(&source) {
            return true;
        }
        if self.adj[&source].contains_key(&target) {
            return false;
        } else {
            return true;
        }
        // No need to check the other way around as edges are undirected
    }
}
