﻿use graph::{linked::Graph, Operator};
use std::fmt;

pub struct Mutant {
    graph: Graph,
    score: f32,
}

impl Mutant {
    #[inline]
    pub const fn new(graph: Graph) -> Self {
        Self { graph, score: 1f32 }
    }
}

pub struct SubGraph<T> {
    mutants: Vec<Mutant>,
    partition_type: T,
}

pub struct Partition<T>(Vec<SubGraph<T>>);
pub struct Mutation<T>(Vec<SubGraph<T>>);
pub struct Rating<T>(Vec<SubGraph<T>>);

pub type PartitionFunc<T> = dyn Fn(Graph) -> Vec<(Graph, T)>;
pub type MutationFunc<T> = dyn Fn(&Graph, &T) -> Vec<Graph>;
pub type RatingFunc = dyn Fn(&Graph) -> f32;

impl<T> Partition<T> {
    pub fn new(g: Graph, f: &PartitionFunc<T>) -> Self {
        let mut ans = Partition::<T>(Default::default());
        for (g, ty) in f(g) {
            ans.0.push(SubGraph {
                mutants: vec![Mutant::new(g)],
                partition_type: ty,
            });
        }
        ans
    }

    #[inline]
    pub fn size(&self) -> Vec<usize> {
        list_size(&self.0)
    }

    pub fn mutate(self, f: &MutationFunc<T>) -> Mutation<T> {
        let mut ans = Mutation::<T>(self.0);
        for sub in &mut ans.0 {
            sub.mutants.extend(
                f(&sub.mutants.first().unwrap().graph, &sub.partition_type)
                    .into_iter()
                    .map(Mutant::new),
            );
        }
        ans
    }
}

impl<T> Mutation<T> {
    #[inline]
    pub fn size(&self) -> Vec<usize> {
        list_size(&self.0)
    }

    pub fn rate(self, f: &RatingFunc) -> Rating<T> {
        let mut ans = Rating::<T>(self.0);
        for sub in ans.0.iter_mut().filter(|sub| sub.mutants.len() > 1) {
            let sum = sub
                .mutants
                .iter_mut()
                .map(|m| -> f32 {
                    m.score = f(&m.graph);
                    m.score
                })
                .sum::<f32>();
            for m in &mut sub.mutants {
                m.score /= sum;
            }
            sub.mutants.sort_by(|a, b| b.score.total_cmp(&a.score))
        }
        ans
    }
}

impl<T> Rating<T> {
    #[inline]
    pub fn size(&self) -> Vec<usize> {
        list_size(&self.0)
    }

    pub fn select(&self, indices: &[usize]) -> Graph {
        debug_assert_eq!(indices.len(), self.0.len());
        let mut ans = Graph::new();
        self.0
            .iter()
            .zip(indices)
            .flat_map(|(sub, idx)| sub.mutants[*idx].graph.ops())
            .for_each(|op| {
                ans.push_op(op.op_type().clone(), op.inputs.clone(), op.outputs.clone());
            });
        ans
    }
}

#[inline]
fn list_size<T>(list: &[SubGraph<T>]) -> Vec<usize> {
    list.iter().map(|sub| sub.mutants.len()).collect()
}

impl<T: fmt::Debug> fmt::Display for SubGraph<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, m) in self.mutants.iter().enumerate() {
            writeln!(
                f,
                "Mutant {i} ({:?}) | Score={} | {}",
                self.partition_type, m.score, m.graph
            )?;
        }
        Ok(())
    }
}

impl<T: fmt::Debug> fmt::Display for Partition<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, sub) in self.0.iter().enumerate() {
            writeln!(
                f,
                "\
Part {i} {{

{sub}
}} // Part {i}"
            )?;
        }
        Ok(())
    }
}
impl<T: fmt::Debug> fmt::Display for Mutation<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, sub) in self.0.iter().enumerate() {
            writeln!(
                f,
                "\
Part {i} {{

{sub}
}} // Part {i}"
            )?;
        }
        Ok(())
    }
}
impl<T: fmt::Debug> fmt::Display for Rating<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, sub) in self.0.iter().enumerate() {
            writeln!(
                f,
                "\
Part {i} {{

{sub}
}} // Part {i}"
            )?;
        }
        Ok(())
    }
}
