"""CLIFF route for Category Theory for AGI course demos."""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .repo_layout import (
    resolve_book_pdf_path,
    resolve_course_repo_root,
    resolve_functorflow_julia_root,
    resolve_julia_examples_root,
)


@dataclass(frozen=True)
class CourseDemoSpec:
    demo_id: str
    title: str
    description: str
    notebook_relpath: str
    colab_url: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class JuliaDemoSpec:
    demo_id: str
    title: str
    description: str
    source_relpath: str
    repo_kind: str
    execution_mode: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]
    inline_script: str = ""


def _course_demo(
    demo_id: str,
    title: str,
    description: str,
    notebook_relpath: str,
    *,
    aliases: tuple[str, ...],
    keywords: tuple[str, ...],
) -> CourseDemoSpec:
    return CourseDemoSpec(
        demo_id=demo_id,
        title=title,
        description=description,
        notebook_relpath=notebook_relpath,
        colab_url=(
            "https://colab.research.google.com/github/sridharmahadevan/"
            f"Category-Theory-for-AGI-UMass-CMPSCI-692CT/blob/main/{notebook_relpath}"
        ),
        aliases=aliases,
        keywords=keywords,
    )


def _julia_demo(
    demo_id: str,
    title: str,
    description: str,
    source_relpath: str,
    *,
    repo_kind: str,
    execution_mode: str,
    aliases: tuple[str, ...],
    keywords: tuple[str, ...],
    inline_script: str = "",
) -> JuliaDemoSpec:
    return JuliaDemoSpec(
        demo_id=demo_id,
        title=title,
        description=description,
        source_relpath=source_relpath,
        repo_kind=repo_kind,
        execution_mode=execution_mode,
        aliases=aliases,
        keywords=keywords,
        inline_script=inline_script,
    )


def _book_section(
    section_id: str,
    title: str,
    description: str,
    start_page: int,
    *,
    aliases: tuple[str, ...],
    keywords: tuple[str, ...],
) -> BookSectionSpec:
    return BookSectionSpec(
        section_id=section_id,
        title=title,
        description=description,
        start_page=start_page,
        aliases=aliases,
        keywords=keywords,
    )


@dataclass(frozen=True)
class CourseTopicGuide:
    topic_id: str
    title: str
    aliases: tuple[str, ...]
    demo_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class CourseProjectIdea:
    title: str
    difficulty: str
    summary: str
    deliverables: tuple[str, ...]
    stretch_goal: str = ""


@dataclass(frozen=True)
class CourseProjectGuide:
    topic_id: str
    title: str
    aliases: tuple[str, ...]
    starter_demo_id: str
    book_section_ids: tuple[str, ...]
    rationale: str
    ideas: tuple[CourseProjectIdea, ...]


@dataclass(frozen=True)
class CourseCodeSnippet:
    snippet_id: str
    topic_id: str
    title: str
    language: str
    description: str
    source_relpath: str
    repo_kind: str
    snippet: str
    follow_up: str = ""
    source_href: str = ""


@dataclass(frozen=True)
class BookSectionSpec:
    section_id: str
    title: str
    description: str
    start_page: int
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class CourseDemoQueryPlan:
    query: str
    normalized_query: str
    explanation_focus: str
    matched_demo_id: str = ""
    matched_title: str = ""
    notebook_path: Path | None = None


@dataclass(frozen=True)
class CourseDemoAgenticConfig:
    query: str
    outdir: Path
    course_repo_root: Path | None = None
    julia_repo_root: Path | None = None
    julia_examples_root: Path | None = None
    book_pdf_path: Path | None = None
    execute_demo: bool = True
    execution_timeout_sec: int = 900

    def resolved(self) -> "CourseDemoAgenticConfig":
        resolved_root = self.course_repo_root.resolve() if self.course_repo_root else _default_course_repo_root()
        resolved_julia_root = self.julia_repo_root.resolve() if self.julia_repo_root else _default_julia_repo_root()
        resolved_julia_examples = (
            self.julia_examples_root.resolve() if self.julia_examples_root else _default_julia_examples_root()
        )
        resolved_book_pdf = self.book_pdf_path.resolve() if self.book_pdf_path else _default_book_pdf_path()
        return CourseDemoAgenticConfig(
            query=" ".join(self.query.split()),
            outdir=self.outdir.resolve(),
            course_repo_root=resolved_root,
            julia_repo_root=resolved_julia_root,
            julia_examples_root=resolved_julia_examples,
            book_pdf_path=resolved_book_pdf,
            execute_demo=bool(self.execute_demo),
            execution_timeout_sec=max(30, int(self.execution_timeout_sec)),
        )


@dataclass(frozen=True)
class CourseDemoRunResult:
    query_plan: CourseDemoQueryPlan
    selected_demo: CourseDemoSpec | JuliaDemoSpec | None
    route_outdir: Path
    notebook_path: Path | None
    generated_script_path: Path | None
    dashboard_path: Path
    summary_path: Path
    response_mode: str
    execution_attempted: bool
    execution_status: str
    stdout_path: Path
    stderr_path: Path
    implementation_language: str = "python"
    selected_demo_source_href: str = ""
    book_pdf_path: Path | None = None
    book_recommendations: tuple[BookSectionSpec, ...] = ()
    book_rationale: str = ""
    code_snippets: tuple[CourseCodeSnippet, ...] = ()
    project_ideas: tuple[CourseProjectIdea, ...] = ()
    project_topic: str = ""
    project_rationale: str = ""
    recommendation_demos: tuple[CourseDemoSpec, ...] = ()
    recommendation_julia_demos: tuple[JuliaDemoSpec, ...] = ()
    recommendation_topic: str = ""
    execution_returncode: int | None = None
    execution_seconds: float | None = None
    error_message: str = ""


_COURSE_DEMOS: tuple[CourseDemoSpec, ...] = (
    _course_demo(
        demo_id="geometric_transformer_sudoku",
        title="Geometric Transformer on Sudoku",
        description="Compare a baseline Transformer and a Geometric Transformer on tiny 4x4 Sudoku with diagrammatic backprop triangles.",
        notebook_relpath="notebooks/week01_sudoku_gt_db.ipynb",
        aliases=("sudoku", "gt sudoku", "geometric transformer sudoku", "diagrammatic backprop"),
        keywords=("sudoku", "geometric transformer", "gt", "diagrammatic backprop"),
    ),
    _course_demo(
        demo_id="geometric_transformer_language_modeling",
        title="Geometric Transformer on Language Modeling",
        description="Compare a baseline Transformer and a GT-Lite language model on a tiny Penn Treebank setup.",
        notebook_relpath="notebooks/week01_lm_gt_vs_transformer.ipynb",
        aliases=("language modeling gt", "gt language modeling", "ptb gt", "gt vs transformer ptb"),
        keywords=("language modeling", "ptb", "penn treebank", "geometric transformer", "gt-lite"),
    ),
    _course_demo(
        demo_id="kan_extension_transformer",
        title="Kan Extension Transformer",
        description="Demonstrate left-Kan aggregation, right-Kan readout, and degeneracy diagnostics for a Kan Extension Transformer.",
        notebook_relpath="notebooks/week05_kan_extension_transformer_demo.ipynb",
        aliases=("kan extension transformer", "ket", "kan transformer", "kan extension"),
        keywords=("kan extension", "ket", "language modeling", "sequence modeling"),
    ),
    _course_demo(
        demo_id="gt_full_ptb",
        title="GT-Full on PTB",
        description="Run the GT-Full tiny PTB language-model comparison with relation-aware message passing.",
        notebook_relpath="notebooks/week06_gt_full_ptb.ipynb",
        aliases=("gt full ptb", "gt-full", "gt full language modeling"),
        keywords=("gt-full", "ptb", "language modeling", "relation-aware"),
    ),
    _course_demo(
        demo_id="backprop_as_functor",
        title="Backprop as a Functor",
        description="Functorial view of backpropagation with a small runnable micro-demo.",
        notebook_relpath="notebooks/week01_backprop_as_functor.ipynb",
        aliases=("backprop as a functor", "functorial backprop", "backprop functor"),
        keywords=("backprop", "functor", "gradient"),
    ),
    _course_demo(
        demo_id="lm_causal_leakage_check",
        title="Language Modeling Causal Leakage Check",
        description="Compare causal and non-causal Transformer/GT variants on a synthetic next-token leakage check.",
        notebook_relpath="notebooks/week01_lm_causal_leakage_check.ipynb",
        aliases=("causal leakage", "leakage check", "future token leakage"),
        keywords=("causal", "leakage", "language modeling", "transformer", "gt"),
    ),
    _course_demo(
        demo_id="coalgebra_demo",
        title="Coalgebra Demo",
        description="Show deterministic systems and MDPs as coalgebras.",
        notebook_relpath="notebooks/week01_coalgebra_demo.ipynb",
        aliases=("coalgebra demo", "coalgebra", "mdp coalgebra"),
        keywords=("coalgebra", "deterministic system", "mdp", "state transition"),
    ),
    _course_demo(
        demo_id="yoneda_microdemo",
        title="Yoneda Micro-Demo",
        description="A small concrete introduction to Yoneda in the course notebook style.",
        notebook_relpath="notebooks/week02_yoneda_microdemo.ipynb",
        aliases=("yoneda microdemo", "yoneda micro-demo", "yoneda lemma demo"),
        keywords=("yoneda", "yoneda lemma"),
    ),
    _course_demo(
        demo_id="yoneda_self_attention",
        title="Yoneda Self-Attention Demo",
        description="Relate Yoneda-style constructions to self-attention behavior.",
        notebook_relpath="notebooks/week02_yoneda_self_attention_demo1.ipynb",
        aliases=("yoneda self attention", "yoneda attention", "self-attention yoneda"),
        keywords=("yoneda", "self-attention", "attention"),
    ),
    _course_demo(
        demo_id="limits_colimits_microdemo",
        title="Limits and Colimits Micro-Demo",
        description="Small runnable examples for limits and colimits.",
        notebook_relpath="notebooks/week03_limits_colimits_microdemo.ipynb",
        aliases=("limits and colimits", "limits colimits", "colimit microdemo"),
        keywords=("limit", "limits", "colimit", "colimits"),
    ),
    _course_demo(
        demo_id="db_colimit_energy",
        title="DB as Colimit Energy",
        description="Use consistency and gluing energy to approximate limits and colimits.",
        notebook_relpath="notebooks/week03_db_colimit_energy.ipynb",
        aliases=("db colimit energy", "colimit energy", "database colimit"),
        keywords=("db", "database", "colimit", "energy", "gluing"),
    ),
    _course_demo(
        demo_id="synthetic_av_pullback",
        title="Synthetic AV Pullback Alignment",
        description="Compare hard and soft pullback constructions for synthetic audio-visual alignment.",
        notebook_relpath="notebooks/week3_synthetic_av_pullback-2.ipynb",
        aliases=("synthetic av pullback", "pullback alignment", "audio visual pullback"),
        keywords=("pullback", "alignment", "audio", "visual", "av"),
    ),
    _course_demo(
        demo_id="clustering_as_functor",
        title="Clustering as a Functor",
        description="Interpret clustering through a functorial lens.",
        notebook_relpath="notebooks/week04_clustering_as_functor.ipynb",
        aliases=("clustering as a functor", "functorial clustering"),
        keywords=("clustering", "functor"),
    ),
    _course_demo(
        demo_id="gt_full_node_labeling",
        title="GT-Full Toy Node Labeling",
        description="Relation-aware GT-Full message passing versus a baseline MLP on node labeling.",
        notebook_relpath="notebooks/week04_gt_full_node_labeling.ipynb",
        aliases=("gt full node labeling", "node labeling", "gt-full node labeling"),
        keywords=("gt-full", "node labeling", "message passing", "graph"),
    ),
    _course_demo(
        demo_id="gt_causal_regimes_ptb",
        title="GT Causal Regimes on PTB",
        description="Compare causal and non-causal GT/Transformer language-model regimes on PTB or WikiText-2.",
        notebook_relpath="notebooks/week04_gt_causal_regimes_ptb_colab.ipynb",
        aliases=("gt causal regimes", "causal regimes ptb", "ptb causal regimes"),
        keywords=("causal regimes", "ptb", "wikitext", "language modeling", "transition metric"),
    ),
    _course_demo(
        demo_id="mini_democritus",
        title="Mini-Democritus",
        description="Map causal triples to GT-Full embeddings and a 2D manifold.",
        notebook_relpath="notebooks/week05_mini_democritus.ipynb",
        aliases=("mini democritus", "democritus mini", "causal manifold"),
        keywords=("democritus", "manifold", "causal triples", "embedding"),
    ),
    _course_demo(
        demo_id="attention_ket",
        title="Attention KET",
        description="Attention formulation with Kan Extension Transformer structure and comparisons.",
        notebook_relpath="notebooks/week05_attention_ket.ipynb",
        aliases=("attention ket", "ket attention"),
        keywords=("ket", "attention", "kan extension"),
    ),
    _course_demo(
        demo_id="causal_attention_ket",
        title="Causal Attention KET",
        description="Causal attention variant of the Kan Extension Transformer architecture.",
        notebook_relpath="notebooks/week05_causal_attention_ket.ipynb",
        aliases=("causal attention ket", "causal ket", "causal kan extension transformer"),
        keywords=("causal", "ket", "attention", "kan extension"),
    ),
    _course_demo(
        demo_id="sheaves_covers",
        title="Sheaves via Covers and Gluing",
        description="A local-covers and overlap-consistency notebook for sheaf-style reasoning.",
        notebook_relpath="notebooks/week07_sheaves_covers.ipynb",
        aliases=("sheaves via covers and gluing", "sheaves covers", "covers and gluing"),
        keywords=("sheaf", "sheaves", "covers", "gluing"),
    ),
    _course_demo(
        demo_id="topos_overlap_penalty",
        title="Topos Overlap Penalty",
        description="A sheaf-style overlap penalty demo that prefers the true causal graph.",
        notebook_relpath="notebooks/week08_topos_overlap_penalty.ipynb",
        aliases=("topos overlap penalty", "overlap penalty", "sheaf overlap"),
        keywords=("topos", "overlap", "penalty", "sheaf", "causal graph"),
    ),
    _course_demo(
        demo_id="causal_discovery_toy",
        title="Toy Causal Discovery",
        description="Recover causal edges from a linear SEM using simple regressions.",
        notebook_relpath="notebooks/week09_causal_discovery_toy.ipynb",
        aliases=("toy causal discovery", "causal discovery toy"),
        keywords=("causal discovery", "sem", "regression"),
    ),
    _course_demo(
        demo_id="kan_do_rn",
        title="Kan-Do Calculus with RN Ratios",
        description="Conditioning versus intervention through Radon-Nikodym reweighting.",
        notebook_relpath="notebooks/week10_kan_do_rn.ipynb",
        aliases=("kan-do calculus", "radon nikodym", "rn ratios"),
        keywords=("kan-do", "radon-nikodym", "conditioning", "intervention"),
    ),
    _course_demo(
        demo_id="subobject_classifier",
        title="Subobject Classifier",
        description="Concrete subobject classifier examples in Sets.",
        notebook_relpath="notebooks/week11_subobject_classifier.ipynb",
        aliases=("subobject classifier", "classifier in sets"),
        keywords=("subobject classifier", "sets", "characteristic map"),
    ),
    _course_demo(
        demo_id="coalgebra_gtdb_rl",
        title="Coalgebraic RL: GT + DB",
        description="Compare MLP, GT, and GT+DB on a stochastic synthetic RL problem.",
        notebook_relpath="notebooks/week11_coalgebra_gtdb_rl_demo.ipynb",
        aliases=("coalgebraic rl", "gt+db rl", "coalgebra rl"),
        keywords=("coalgebra", "rl", "gt+db", "stochastic transitions"),
    ),
    _course_demo(
        demo_id="jstability_regimes",
        title="j-Stability Across Regimes",
        description="Find stable edges across environments in synthetic SEM regimes.",
        notebook_relpath="notebooks/week12_jstability_regimes.ipynb",
        aliases=("j-stability", "j stability", "stability across regimes"),
        keywords=("j-stability", "stability", "regimes", "sem"),
    ),
    _course_demo(
        demo_id="democritus_manifold",
        title="Democritus Causal Manifold",
        description="Visualize a lightweight Democritus-style topic manifold.",
        notebook_relpath="notebooks/week13_democritus_manifold.ipynb",
        aliases=("democritus causal manifold", "democritus manifold", "topic manifold"),
        keywords=("democritus", "manifold", "topic manifold"),
    ),
    _course_demo(
        demo_id="open_games_attention_economy",
        title="Open Games: Attention Economy",
        description="A simple open-games demo for attention competition and co-play.",
        notebook_relpath="notebooks/week13_open_games_attention_economy.ipynb",
        aliases=("open games", "attention economy", "open games attention economy"),
        keywords=("open games", "attention economy", "game"),
    ),
    _course_demo(
        demo_id="comparing_gt_vs_umap",
        title="GT+DB vs UMAP",
        description="Compare GT+DB relational embeddings to UMAP on graph projections.",
        notebook_relpath="notebooks/Comparing_GT_vs_UMAP.ipynb",
        aliases=("gt vs umap", "gt+db vs umap", "comparing gt vs umap"),
        keywords=("umap", "gt+db", "relational domains", "embedding"),
    ),
    _course_demo(
        demo_id="comparing_gt_vs_umap_dblp",
        title="GT+DB vs UMAP on DBLP",
        description="DBLP benchmark contrasting triangle-aware GT embeddings with UMAP on the 1-skeleton.",
        notebook_relpath="notebooks/Comparing_GT_vs_UMAP_DBLP.ipynb",
        aliases=("gt vs umap dblp", "dblp", "umap dblp"),
        keywords=("dblp", "umap", "gt+db", "heterogeneous graph"),
    ),
)

_COURSE_DEMOS_BY_ID = {demo.demo_id: demo for demo in _COURSE_DEMOS}

_BOOK_SECTIONS: tuple[BookSectionSpec, ...] = (
    _book_section(
        section_id="categorical_deep_learning",
        title="Categorical Deep Learning",
        description="High-level framing for how category-theoretic ideas organize modern learning architectures.",
        start_page=65,
        aliases=("categorical deep learning", "deep learning"),
        keywords=("category theory", "deep learning", "architecture"),
    ),
    _book_section(
        section_id="diagrammatic_backpropagation",
        title="Diagrammatic Backpropagation",
        description="Introduces the triangle-based diagrammatic backpropagation view used in the early GT demos.",
        start_page=75,
        aliases=("diagrammatic backpropagation", "diagrammatic backprop", "backpropagation"),
        keywords=("diagrammatic backprop", "backprop", "triangle"),
    ),
    _book_section(
        section_id="geometric_transformers",
        title="Geometric Transformers",
        description="Main textbook chapter for the GT construction, including the intuition behind relation-aware updates.",
        start_page=83,
        aliases=("geometric transformers", "geometric transformer", "gt"),
        keywords=("geometric transformer", "gt", "sudoku", "relation aware"),
    ),
    _book_section(
        section_id="information_regimes_gt",
        title="Information Regimes in Geometric Transformers",
        description="Explains the information-flow regimes behind causal and non-causal GT variants.",
        start_page=133,
        aliases=("information regimes in geometric transformers", "information regimes", "causal regimes"),
        keywords=("information regimes", "causal regimes", "gt causal"),
    ),
    _book_section(
        section_id="kan_extension_transformers",
        title="Kan Extension and Topological Coend Transformers",
        description="Primary reference chapter for KET-style models, Kan aggregation, and related transformer constructions.",
        start_page=143,
        aliases=("kan extension and topological coend transformers", "kan extension transformers", "kan extension transformer", "ket"),
        keywords=("kan extension", "ket", "topological coend", "transformer"),
    ),
    _book_section(
        section_id="structured_language_modeling",
        title="Structured Language Modeling",
        description="Connects transformer constructions to concrete language-modeling tasks and evaluations.",
        start_page=179,
        aliases=("structured language modeling", "language modeling"),
        keywords=("language modeling", "ptb", "sequence modeling"),
    ),
    _book_section(
        section_id="manifold_learning_gt",
        title="Manifold Learning with Geometric Transformers",
        description="Textbook treatment of manifold learning and relational embedding ideas behind Democritus-style demos.",
        start_page=195,
        aliases=("manifold learning with geometric transformers", "manifold learning", "democritus manifold"),
        keywords=("manifold", "embedding", "democritus"),
    ),
    _book_section(
        section_id="causality_from_language",
        title="Causality from Language",
        description="Introduces the book's causal lens, especially causal discovery and language-mediated structure.",
        start_page=273,
        aliases=("causality from language", "causality", "causality from text"),
        keywords=("causal", "causality", "causal discovery"),
    ),
    _book_section(
        section_id="agentic_systems_ket",
        title="Building Agentic Systems using Kan Extension Transformers",
        description="Relates KET constructions to agentic systems and more application-oriented workflows.",
        start_page=307,
        aliases=("building agentic systems using kan extension transformers", "agentic systems", "agentic ket"),
        keywords=("agentic", "kan extension", "ket", "agents"),
    ),
    _book_section(
        section_id="topos_causal_models",
        title="Topos Causal Models",
        description="Main chapter for topos-style causal semantics, overlap penalties, and structural reasoning.",
        start_page=321,
        aliases=("topos causal models", "topos", "sheaves and topos"),
        keywords=("topos", "causal graph", "sheaf"),
    ),
    _book_section(
        section_id="judo_calculus",
        title="Judo Calculus",
        description="Book chapter for intervention/conditioning distinctions and the calculus around them.",
        start_page=335,
        aliases=("judo calculus", "do calculus", "kan do"),
        keywords=("intervention", "conditioning", "radon nikodym", "kan do"),
    ),
    _book_section(
        section_id="universal_reinforcement_learning",
        title="Universal Reinforcement Learning",
        description="Main textbook chapter for the RL direction and its categorical interpretation.",
        start_page=455,
        aliases=("universal reinforcement learning", "reinforcement learning", "rl"),
        keywords=("reinforcement learning", "rl", "coalgebra"),
    ),
    _book_section(
        section_id="deep_url_gt",
        title="Deep URL with Geometric Transformers",
        description="Extends the universal RL story with Geometric Transformers in the deep-learning setting.",
        start_page=485,
        aliases=("deep url with geometric transformers", "deep url"),
        keywords=("deep url", "gt", "rl"),
    ),
    _book_section(
        section_id="code_companion",
        title="Code Companion",
        description="Pointers from the book into runnable code and notebooks for hands-on exploration.",
        start_page=521,
        aliases=("code companion",),
        keywords=("code", "companion", "notebooks"),
    ),
)

_BOOK_SECTIONS_BY_ID = {section.section_id: section for section in _BOOK_SECTIONS}

_DEMO_TO_BOOK_SECTIONS: dict[str, tuple[str, ...]] = {
    "geometric_transformer_sudoku": ("geometric_transformers", "diagrammatic_backpropagation"),
    "geometric_transformer_language_modeling": ("geometric_transformers", "structured_language_modeling"),
    "kan_extension_transformer": ("kan_extension_transformers", "structured_language_modeling"),
    "gt_full_ptb": ("information_regimes_gt", "structured_language_modeling"),
    "backprop_as_functor": ("categorical_deep_learning", "diagrammatic_backpropagation"),
    "lm_causal_leakage_check": ("information_regimes_gt", "structured_language_modeling"),
    "coalgebra_demo": ("universal_reinforcement_learning",),
    "yoneda_microdemo": ("categorical_deep_learning",),
    "yoneda_self_attention": ("categorical_deep_learning", "kan_extension_transformers"),
    "limits_colimits_microdemo": ("categorical_deep_learning",),
    "db_colimit_energy": ("categorical_deep_learning",),
    "synthetic_av_pullback": ("categorical_deep_learning",),
    "clustering_as_functor": ("categorical_deep_learning",),
    "gt_full_node_labeling": ("geometric_transformers",),
    "gt_causal_regimes_ptb": ("information_regimes_gt", "structured_language_modeling"),
    "mini_democritus": ("manifold_learning_gt",),
    "attention_ket": ("kan_extension_transformers",),
    "causal_attention_ket": ("kan_extension_transformers", "causality_from_language"),
    "sheaves_covers": ("topos_causal_models",),
    "topos_overlap_penalty": ("topos_causal_models", "judo_calculus"),
    "causal_discovery_toy": ("causality_from_language",),
    "kan_do_rn": ("judo_calculus", "topos_causal_models"),
    "subobject_classifier": ("topos_causal_models",),
    "coalgebra_gtdb_rl": ("universal_reinforcement_learning", "deep_url_gt"),
    "jstability_regimes": ("causality_from_language", "topos_causal_models"),
    "democritus_manifold": ("manifold_learning_gt",),
    "open_games_attention_economy": ("agentic_systems_ket",),
    "comparing_gt_vs_umap": ("manifold_learning_gt", "geometric_transformers"),
    "comparing_gt_vs_umap_dblp": ("manifold_learning_gt", "geometric_transformers"),
    "julia_ket_block": ("kan_extension_transformers", "agentic_systems_ket"),
    "julia_causal_semantics": ("causality_from_language", "judo_calculus"),
    "julia_sudoku_gt_lux": ("geometric_transformers", "diagrammatic_backpropagation"),
}

_JULIA_DEMOS: tuple[JuliaDemoSpec, ...] = (
    _julia_demo(
        demo_id="julia_ket_block",
        title="Julia KET Block",
        description="Run a compact FunctorFlow.jl KET example showing left Kan aggregation as the core transformer-like operation.",
        source_relpath="README.md",
        repo_kind="functorflow_jl",
        execution_mode="inline",
        aliases=("julia ket", "julia version of ket", "julia kan extension transformer", "julia ket block"),
        keywords=("julia", "ket", "kan extension", "transformer"),
        inline_script="""
using Pkg
Pkg.activate("/Users/sridharmahadevan/Documents/Playground/FunctorFlow.jl"; io=devnull)
using FunctorFlow

D = Diagram(:MyKET)
add_object!(D, :Values; kind=:messages)
add_object!(D, :Incidence; kind=:relation)
add_left_kan!(D, :aggregate; source=:Values, along=:Incidence, reducer=:sum)

compiled = compile_to_callable(D)
result = FunctorFlow.run(compiled, Dict(
    :Values => Dict(1 => 1.0, 2 => 2.0, 3 => 4.0),
    :Incidence => Dict("left" => [1, 2], "right" => [2, 3])
))

println(D)
println("KET aggregation (sum): ", result.values[:aggregate])
""".strip(),
    ),
    _julia_demo(
        demo_id="julia_causal_semantics",
        title="Julia Causal Semantics",
        description="Run a compact FunctorFlow.jl causal-semantics example with intervention and conditioning as left and right Kan extensions.",
        source_relpath="vignettes/09-causal-semantics/causal-semantics.qmd",
        repo_kind="functorflow_jl",
        execution_mode="inline",
        aliases=("julia causality", "julia causal semantics", "julia do calculus", "julia intervention"),
        keywords=("julia", "causal", "causality", "intervention", "conditioning"),
        inline_script="""
using Pkg
Pkg.activate("/Users/sridharmahadevan/Documents/Playground/FunctorFlow.jl"; io=devnull)
using FunctorFlow

ctx = CausalContext(:SmokingContext; observational_regime=:obs, interventional_regime=:do)
cd = build_causal_diagram(:SmokingCancer; context=ctx)
obs_data = Dict(
    :Observations => Dict(:a => 1.0, :b => 2.0, :c => 3.0),
    :CausalStructure => Dict((:a, :b) => true, (:b, :c) => true)
)
result = interventional_expectation(cd, obs_data)

println("Causal diagram: ", cd.name)
println("Intervention Kan: ", cd.intervention_kan)
println("Conditioning Kan: ", cd.conditioning_kan)
println("Intervention result: ", result[:intervention])
println("Conditioning result: ", result[:conditioning])
""".strip(),
    ),
    _julia_demo(
        demo_id="julia_sudoku_gt_lux",
        title="Julia Sudoku GT with Lux",
        description="Run the Julia FunctorFlow Sudoku GT example using Lux-backed KET attention and Sudoku constraint structure.",
        source_relpath="sudoku_gt_lux.jl",
        repo_kind="julia_ff",
        execution_mode="script",
        aliases=("julia sudoku gt", "julia geometric transformer sudoku", "julia sudoku"),
        keywords=("julia", "sudoku", "gt", "lux", "ket"),
    ),
)

_JULIA_DEMOS_BY_ID = {demo.demo_id: demo for demo in _JULIA_DEMOS}

_RECOMMENDATION_CUES = (
    "what demo",
    "which demo",
    "which julia demo",
    "where should i start",
    "how should i explore",
    "help me explore",
    "suggest a demo",
    "suggest demos",
    "recommend a demo",
    "recommend demos",
    "what should i look at",
    "what should i use",
    "which notebook",
    "which notebooks",
)

_PROJECT_CUES = (
    "project suggestion",
    "project idea",
    "project ideas",
    "suggest a project",
    "suggest project",
    "recommend a project",
    "recommend project",
    "final project",
    "course project",
    "student project",
    "reader project",
)

_LEARNING_CUES = (
    "learn about",
    "teach me",
    "explain how",
    "show code",
    "show me the code",
    "code snippet",
    "implementation snippet",
    "implementation",
    "how is it implemented",
    "how does the code work",
    "how it works",
    "how this works",
)

_COURSE_TOPIC_GUIDES: tuple[CourseTopicGuide, ...] = (
    CourseTopicGuide(
        topic_id="causality",
        title="Causality",
        aliases=("causality", "causal", "intervention", "counterfactual", "causal inference"),
        demo_ids=("causal_discovery_toy", "kan_do_rn", "topos_overlap_penalty", "gt_causal_regimes_ptb"),
        rationale="These demos move from simple causal recovery to intervention semantics and structure-sensitive causal regularization.",
    ),
    CourseTopicGuide(
        topic_id="language_modeling",
        title="Language Modeling",
        aliases=("language modeling", "language model", "lm", "ptb", "wikitext", "sequence modeling"),
        demo_ids=("geometric_transformer_language_modeling", "kan_extension_transformer", "gt_causal_regimes_ptb", "gt_full_ptb"),
        rationale="These notebooks compare Transformer-style sequence models through increasingly categorical and relation-aware constructions.",
    ),
    CourseTopicGuide(
        topic_id="sheaves_topos",
        title="Sheaves and Topos",
        aliases=("sheaf", "sheaves", "topos", "gluing", "covers"),
        demo_ids=("sheaves_covers", "topos_overlap_penalty", "subobject_classifier"),
        rationale="This path starts with local covers and gluing, then moves to overlap consistency and a concrete topos-flavored classifier construction.",
    ),
    CourseTopicGuide(
        topic_id="democritus_manifolds",
        title="Democritus and Manifolds",
        aliases=("democritus", "manifold", "topic manifold", "causal manifold"),
        demo_ids=("mini_democritus", "democritus_manifold", "comparing_gt_vs_umap"),
        rationale="These demos show the progression from a tiny causal manifold construction to larger relational embedding comparisons.",
    ),
    CourseTopicGuide(
        topic_id="coalgebra_rl",
        title="Coalgebra and RL",
        aliases=("coalgebra", "rl", "reinforcement learning", "mdp"),
        demo_ids=("coalgebra_demo", "coalgebra_gtdb_rl"),
        rationale="Start with the coalgebraic framing itself, then move to the GT versus GT+DB reinforcement-learning comparison.",
    ),
    CourseTopicGuide(
        topic_id="attention_yoneda",
        title="Attention and Yoneda",
        aliases=("attention", "yoneda", "self attention", "self-attention"),
        demo_ids=("yoneda_microdemo", "yoneda_self_attention", "attention_ket", "causal_attention_ket"),
        rationale="These notebooks connect Yoneda intuitions to attention, then extend them into KET-style constructions.",
    ),
    CourseTopicGuide(
        topic_id="limits_colimits",
        title="Limits, Colimits, and Pullbacks",
        aliases=("limit", "limits", "colimit", "colimits", "pullback"),
        demo_ids=("limits_colimits_microdemo", "db_colimit_energy", "synthetic_av_pullback"),
        rationale="This set moves from basic categorical objects to gluing-energy and pullback alignment examples.",
    ),
)

_JULIA_TOPIC_GUIDES: tuple[CourseTopicGuide, ...] = (
    CourseTopicGuide(
        topic_id="julia_ket",
        title="Julia KET and Kan Extensions",
        aliases=("julia ket", "julia kan extension", "julia transformer", "julia version", "ket", "kan extension"),
        demo_ids=("julia_ket_block", "julia_sudoku_gt_lux"),
        rationale="These Julia demos show both the minimal KET block and a richer Sudoku GT construction built with FunctorFlow.jl and Lux.",
    ),
    CourseTopicGuide(
        topic_id="julia_causality",
        title="Julia Causality",
        aliases=("julia causality", "julia causal", "julia intervention", "julia do calculus"),
        demo_ids=("julia_causal_semantics",),
        rationale="This Julia path uses FunctorFlow.jl's causal-semantics machinery to expose intervention and conditioning as Kan extensions.",
    ),
)

_COURSE_PROJECT_GUIDES: tuple[CourseProjectGuide, ...] = (
    CourseProjectGuide(
        topic_id="kan_extension_transformers",
        title="Kan Extension Transformers",
        aliases=("kan extension transformer", "kan extension", "ket", "kan transformers"),
        starter_demo_id="kan_extension_transformer",
        book_section_ids=("kan_extension_transformers", "agentic_systems_ket", "structured_language_modeling"),
        rationale="A strong KET project usually starts with the core Kan aggregation demo, then moves into either structured sequence modeling or an agentic graph setting.",
        ideas=(
            CourseProjectIdea(
                title="KET vs Transformer on a structured sequence toy task",
                difficulty="beginner",
                summary="Compare a baseline Transformer and a Kan Extension Transformer on a small synthetic sequence task where typed relations between tokens are explicit.",
                deliverables=("run the KET demo", "define a synthetic structured dataset", "compare accuracy and failure cases", "write a short report"),
                stretch_goal="Add an ablation showing what changes when the relation structure is removed.",
            ),
            CourseProjectIdea(
                title="Structured language modeling with explicit relation maps",
                difficulty="intermediate",
                summary="Adapt the KET setup to a language-modeling problem in which token neighborhoods or relation maps are part of the model input.",
                deliverables=("start from the language-modeling notebooks", "design relation-aware inputs", "benchmark against a baseline Transformer", "analyze qualitative errors"),
                stretch_goal="Measure when KET helps most as relation density or sequence length changes.",
            ),
            CourseProjectIdea(
                title="Agentic memory or retrieval with KET aggregation",
                difficulty="advanced",
                summary="Use KET as the aggregation layer over a task graph or memory graph in a lightweight agentic system.",
                deliverables=("define a graph-structured memory task", "build a KET-based aggregator", "compare against a simpler retrieval baseline", "summarize design lessons"),
                stretch_goal="Implement both Python and Julia variants and compare how the abstractions line up.",
            ),
        ),
    ),
    CourseProjectGuide(
        topic_id="geometric_transformers",
        title="Geometric Transformers",
        aliases=("geometric transformer", "geometric transformers", "gt", "gt-full"),
        starter_demo_id="geometric_transformer_sudoku",
        book_section_ids=("geometric_transformers", "diagrammatic_backpropagation", "structured_language_modeling"),
        rationale="GT projects work well when students begin with a concrete relation-aware demo, then scale to a richer domain like graphs or language.",
        ideas=(
            CourseProjectIdea(
                title="Geometric Transformer on a new constraint satisfaction task",
                difficulty="beginner",
                summary="Port the Sudoku setup to another toy structured problem and study how GT updates use the relation graph.",
                deliverables=("run the Sudoku GT notebook", "define a new task", "visualize performance against a baseline", "document the relation structure"),
                stretch_goal="Add a diagrammatic backprop visualization for the new task.",
            ),
            CourseProjectIdea(
                title="GT-Full for relational node labeling",
                difficulty="intermediate",
                summary="Use the GT-Full machinery on a graph labeling task and examine how relation-aware message passing changes errors.",
                deliverables=("start from the GT-Full node-labeling demo", "create a clear evaluation protocol", "compare with an MLP baseline", "analyze edge cases"),
                stretch_goal="Study how performance changes as graph sparsity or noise varies.",
            ),
        ),
    ),
    CourseProjectGuide(
        topic_id="causality",
        title="Causality",
        aliases=("causality", "causal", "intervention", "counterfactual", "causal discovery"),
        starter_demo_id="causal_discovery_toy",
        book_section_ids=("causality_from_language", "topos_causal_models", "judo_calculus"),
        rationale="Causality projects can start with a tractable causal discovery notebook and then branch into interventions, regime changes, or structural regularization.",
        ideas=(
            CourseProjectIdea(
                title="Causal discovery under noisy interventions",
                difficulty="beginner",
                summary="Extend the toy causal discovery setup with noisy or partial interventions and measure robustness.",
                deliverables=("run the toy discovery demo", "add noisy intervention settings", "compare recovered graphs", "write up what fails first"),
                stretch_goal="Relate the results to the Judo Calculus chapter.",
            ),
            CourseProjectIdea(
                title="Topos-style overlap penalties for causal graph selection",
                difficulty="advanced",
                summary="Use the overlap-penalty or sheaf-style causal demos to study whether structural consistency helps recover the right graph.",
                deliverables=("run the topos overlap demo", "define a family of graph ambiguities", "evaluate structural penalties", "summarize insights"),
                stretch_goal="Combine the causal and language-modeling routes into a shared causal representation experiment.",
            ),
        ),
    ),
    CourseProjectGuide(
        topic_id="sheaves_topos",
        title="Sheaves and Topos",
        aliases=("sheaf", "sheaves", "topos", "gluing", "covers"),
        starter_demo_id="sheaves_covers",
        book_section_ids=("topos_causal_models", "judo_calculus"),
        rationale="These projects are good for students who want a mathematically explicit build: local covers, overlap consistency, and gluing all produce tangible experiments.",
        ideas=(
            CourseProjectIdea(
                title="Local consistency and gluing on a multimodal toy dataset",
                difficulty="intermediate",
                summary="Model local views of a toy multimodal dataset and study how gluing constraints affect prediction or reconstruction.",
                deliverables=("run the sheaves covers demo", "create local-view subsets", "define an overlap-consistency metric", "report how gluing changes outcomes"),
                stretch_goal="Compare a sheaf-inspired penalty to a plain regularization baseline.",
            ),
        ),
    ),
    CourseProjectGuide(
        topic_id="democritus_manifolds",
        title="Democritus and Manifolds",
        aliases=("democritus", "manifold", "topic manifold", "causal manifold"),
        starter_demo_id="democritus_manifold",
        book_section_ids=("manifold_learning_gt", "geometric_transformers"),
        rationale="This direction is ideal for students who want to connect latent geometry, topic structure, and relational embeddings.",
        ideas=(
            CourseProjectIdea(
                title="Topic or concept manifolds from relational corpora",
                difficulty="intermediate",
                summary="Build a small relational corpus and compare manifold structure learned by GT-style embeddings against a simpler baseline.",
                deliverables=("run the Democritus manifold demo", "define a small corpus or graph", "visualize embeddings", "interpret neighborhoods and failures"),
                stretch_goal="Compare GT+DB against UMAP or another non-relational projection method.",
            ),
        ),
    ),
    CourseProjectGuide(
        topic_id="coalgebra_rl",
        title="Coalgebra and RL",
        aliases=("coalgebra", "rl", "reinforcement learning", "mdp"),
        starter_demo_id="coalgebra_gtdb_rl",
        book_section_ids=("universal_reinforcement_learning", "deep_url_gt"),
        rationale="Coalgebra and RL projects combine a clear formal lens with measurable behavior on sequential decision problems.",
        ideas=(
            CourseProjectIdea(
                title="Coalgebraic RL with structured transition models",
                difficulty="advanced",
                summary="Compare MLP, GT, and GT+DB approaches on a controlled RL problem where transition structure is meaningful.",
                deliverables=("run the coalgebraic RL demo", "define evaluation metrics across regimes", "compare policy or value quality", "write a synthesis of when structure helps"),
                stretch_goal="Test whether the gains persist when the environment shifts across regimes.",
            ),
        ),
    ),
)

_COURSE_CODE_SNIPPETS: tuple[CourseCodeSnippet, ...] = (
    CourseCodeSnippet(
        snippet_id="ket_python_minimal",
        topic_id="kan_extension_transformers",
        title="Kan Extension Transformer core aggregation in PyTorch",
        language="python",
        description="Minimal PyTorch-style left-Kan and right-Kan core from the course KET demo.",
        source_relpath="notebooks/week05_kan_extension_transformer_demo.ipynb",
        repo_kind="course_repo",
        snippet="""def kernel(Eq, Ek, tau):
    sim = torch.matmul(Ek, Eq) / tau
    return torch.exp(sim)

def left_kan_aggregate(Eq, Ek, values, tau):
    w = kernel(Eq, Ek, tau)
    w = w / (w.sum() + 1e-9)
    agg = torch.sum(w.unsqueeze(-1) * values, dim=0)
    return agg, w

def right_kan_readout(Eq, Ek, logits, tau):
    w = kernel(Eq, Ek, tau)
    w = w / (w.sum() + 1e-9)
    logits = torch.sum(w.unsqueeze(-1) * logits, dim=0)
    return logits, w""",
        follow_up="Try swapping the reducer behavior or temperature schedule to see how the aggregation sharpens.",
    ),
    CourseCodeSnippet(
        snippet_id="ket_julia_minimal",
        topic_id="kan_extension_transformers",
        title="Kan Extension Transformer block in Julia/FunctorFlow",
        language="julia",
        description="Minimal FunctorFlow.jl KET block using a left Kan extension over an incidence relation.",
        source_relpath="README.md",
        repo_kind="functorflow_jl",
        snippet="""D = Diagram(:MyKET)
add_object!(D, :Values; kind=:messages)
add_object!(D, :Incidence; kind=:relation)
add_left_kan!(D, :aggregate; source=:Values, along=:Incidence, reducer=:sum)

compiled = compile_to_callable(D)
result = FunctorFlow.run(compiled, Dict(
    :Values => Dict(1 => 1.0, 2 => 2.0, 3 => 4.0),
    :Incidence => Dict("left" => [1, 2], "right" => [2, 3])
))""",
        follow_up="Replace the simple `:sum` reducer with a different aggregation pattern or a learned attention-like layer.",
    ),
    CourseCodeSnippet(
        snippet_id="gt_python_sudoku",
        topic_id="geometric_transformers",
        title="Geometric Transformer neighborhood aggregation sketch",
        language="python",
        description="Compact sketch of relation-aware neighborhood aggregation in the GT course material.",
        source_relpath="notebooks/week01_sudoku_gt_db.ipynb",
        repo_kind="course_repo",
        snippet="""peer_messages = peer_embeddings[peer_index]
peer_weights = torch.softmax(peer_scores, dim=0)
aggregated = torch.sum(peer_weights.unsqueeze(-1) * peer_messages, dim=0)
updated_cell = cell_embedding + aggregated""",
        follow_up="Change the neighborhood definition or add typed edges for rows, columns, and blocks separately.",
    ),
    CourseCodeSnippet(
        snippet_id="causal_julia_semantics",
        topic_id="causality",
        title="Causal semantics with Kan structure in Julia",
        language="julia",
        description="Small Julia fragment showing causal context and intervention-style computation.",
        source_relpath="vignettes/09-causal-semantics/causal-semantics.qmd",
        repo_kind="functorflow_jl",
        snippet="""ctx = CausalContext(:SmokingContext; observational_regime=:obs, interventional_regime=:do)
cd = build_causal_diagram(:SmokingCancer; context=ctx)
result = interventional_expectation(cd, obs_data)

println(\"Intervention Kan: \", cd.intervention_kan)
println(\"Conditioning Kan: \", cd.conditioning_kan)""",
        follow_up="Modify the observational data or the causal structure and compare intervention versus conditioning outputs.",
    ),
)


def _default_course_repo_root() -> Path:
    return resolve_course_repo_root()


def _default_julia_repo_root() -> Path:
    return resolve_functorflow_julia_root()


def _default_julia_examples_root() -> Path:
    return resolve_julia_examples_root()


def _default_book_pdf_path() -> Path:
    return resolve_book_pdf_path()


def _default_julia_depot_path() -> Path:
    return Path(os.environ.get("CLIFF_JULIA_DEPOT_PATH", "/tmp/julia_depot"))


def _resolve_julia_bin() -> str:
    env_value = os.environ.get("CLIFF_JULIA_BIN", "").strip()
    if env_value:
        return env_value
    if shutil.which("julia"):
        return "julia"
    return str(Path.home() / ".juliaup" / "bin" / "julia")


def _resolve_juliaup_bin() -> str:
    env_value = os.environ.get("CLIFF_JULIAUP_BIN", "").strip()
    if env_value:
        return env_value
    if shutil.which("juliaup"):
        return "juliaup"
    return str(Path.home() / ".juliaup" / "bin" / "juliaup")


def looks_like_course_demo_query(query: str) -> bool:
    normalized = _normalize_query_text(query)
    if "category theory for agi" in normalized or "textbook" in normalized:
        return True
    if is_course_project_query(query) and recommend_course_project_ideas(query)[0]:
        return True
    if is_course_learning_query(query) and recommend_course_learning_resources(query)[0]:
        return True
    if is_course_recommendation_query(query) and (recommend_course_demos(query) or recommend_julia_demos(query)):
        return True
    if match_julia_demo(query) is not None:
        return True
    demo = match_course_demo(query)
    return demo is not None


def match_course_demo(query: str) -> CourseDemoSpec | None:
    normalized = _normalize_query_text(query)
    best_demo: CourseDemoSpec | None = None
    best_score = 0
    for demo in _COURSE_DEMOS:
        score = 0
        for alias in demo.aliases:
            if _contains_normalized_phrase(normalized, alias):
                score += 5
        for keyword in demo.keywords:
            if _contains_normalized_phrase(normalized, keyword):
                score += 2
        if demo.demo_id == "geometric_transformer_sudoku" and "sudoku" in normalized:
            score += 4
        if demo.demo_id == "kan_extension_transformer" and "kan extension" in normalized:
            score += 4
        if demo.demo_id == "geometric_transformer_language_modeling" and (
            "language modeling" in normalized or "ptb" in normalized
        ):
            score += 3
        if demo.demo_id == "gt_full_ptb" and "gt full" in normalized:
            score += 4
        if demo.demo_id == "sheaves_covers" and ("sheaf" in normalized or "sheaves" in normalized):
            score += 4
        if demo.demo_id == "democritus_manifold" and ("democritus" in normalized and "manifold" in normalized):
            score += 4
        if demo.demo_id == "open_games_attention_economy" and "open games" in normalized:
            score += 4
        if demo.demo_id == "comparing_gt_vs_umap_dblp" and "dblp" in normalized:
            score += 5
        if demo.demo_id == "comparing_gt_vs_umap" and ("umap" in normalized and "relational" in normalized):
            score += 4
        if demo.demo_id == "coalgebra_gtdb_rl" and ("gt+db" in normalized or "rl" in normalized):
            score += 3
        if demo.demo_id == "subobject_classifier" and "subobject classifier" in normalized:
            score += 4
        if demo.demo_id == "jstability_regimes" and ("j stability" in normalized or "j-stability" in normalized):
            score += 4
        if score > best_score:
            best_demo = demo
            best_score = score
    return best_demo if best_score > 0 else None


def is_course_recommendation_query(query: str) -> bool:
    normalized = _normalize_query_text(query)
    if any(cue in normalized for cue in _RECOMMENDATION_CUES):
        return True
    return "demo" in normalized and any(token in normalized for token in ("which", "what", "recommend", "suggest"))


def is_course_project_query(query: str) -> bool:
    normalized = _normalize_query_text(query)
    if any(cue in normalized for cue in _PROJECT_CUES):
        return True
    return "project" in normalized and any(token in normalized for token in ("suggest", "recommend", "idea", "ideas", "would like"))


def is_course_learning_query(query: str) -> bool:
    normalized = _normalize_query_text(query)
    if any(cue in normalized for cue in _LEARNING_CUES):
        return True
    if (
        normalized.startswith("explain the ")
        and " on " not in normalized
        and " problem" not in normalized
        and " demo" not in normalized
        and " notebook" not in normalized
    ):
        return True
    if "explain" in normalized and "works" in normalized:
        return True
    return "learn" in normalized and any(token in normalized for token in ("kan extension", "ket", "geometric transformer", "causal", "sheaf", "yoneda"))


def recommend_course_demos(query: str, *, limit: int = 3) -> tuple[str, tuple[CourseDemoSpec, ...], str]:
    normalized = _normalize_query_text(query)
    best_guide: CourseTopicGuide | None = None
    best_score = 0
    for guide in _COURSE_TOPIC_GUIDES:
        score = 0
        for alias in guide.aliases:
            if _contains_normalized_phrase(normalized, alias):
                score += 3
        if score > best_score:
            best_guide = guide
            best_score = score
    if best_guide is None or best_score == 0:
        return "", (), ""
    demos = tuple(_COURSE_DEMOS_BY_ID[demo_id] for demo_id in best_guide.demo_ids if demo_id in _COURSE_DEMOS_BY_ID)[:limit]
    return best_guide.title, demos, best_guide.rationale


def match_julia_demo(query: str) -> JuliaDemoSpec | None:
    normalized = _normalize_query_text(query)
    if "julia" not in normalized:
        return None
    best_demo: JuliaDemoSpec | None = None
    best_score = 0
    for demo in _JULIA_DEMOS:
        score = 0
        for alias in demo.aliases:
            if _contains_normalized_phrase(normalized, alias):
                score += 5
        for keyword in demo.keywords:
            if _contains_normalized_phrase(normalized, keyword):
                score += 2
        if demo.demo_id == "julia_ket_block" and ("ket" in normalized or "kan extension" in normalized):
            score += 4
        if demo.demo_id == "julia_causal_semantics" and ("causal" in normalized or "causality" in normalized):
            score += 4
        if demo.demo_id == "julia_sudoku_gt_lux" and "sudoku" in normalized:
            score += 4
        if score > best_score:
            best_demo = demo
            best_score = score
    return best_demo if best_score > 0 else None


def recommend_julia_demos(query: str, *, limit: int = 3) -> tuple[str, tuple[JuliaDemoSpec, ...], str]:
    normalized = _normalize_query_text(query)
    if "julia" not in normalized:
        return "", (), ""
    best_guide: CourseTopicGuide | None = None
    best_score = 0
    for guide in _JULIA_TOPIC_GUIDES:
        score = 0
        for alias in guide.aliases:
            if _contains_normalized_phrase(normalized, alias):
                score += 3
        if guide.topic_id == "julia_ket" and ("ket" in normalized or "kan extension" in normalized):
            score += 4
        if guide.topic_id == "julia_causality" and ("causal" in normalized or "causality" in normalized):
            score += 4
        if score > best_score:
            best_guide = guide
            best_score = score
    if best_guide is None or best_score == 0:
        return "", (), ""
    demos = tuple(_JULIA_DEMOS_BY_ID[demo_id] for demo_id in best_guide.demo_ids if demo_id in _JULIA_DEMOS_BY_ID)[:limit]
    return best_guide.title, demos, best_guide.rationale


def recommend_book_sections(
    query: str,
    *,
    matched_demo_id: str = "",
    topic_title: str = "",
    limit: int = 3,
) -> tuple[tuple[BookSectionSpec, ...], str]:
    normalized = _normalize_query_text(query)
    score_by_section: dict[str, int] = {}
    for section in _BOOK_SECTIONS:
        score = 0
        for alias in section.aliases:
            if _contains_normalized_phrase(normalized, alias):
                score += 5
        for keyword in section.keywords:
            if _contains_normalized_phrase(normalized, keyword):
                score += 2
        score_by_section[section.section_id] = score

    if topic_title:
        topic_normalized = _normalize_query_text(topic_title)
        for section in _BOOK_SECTIONS:
            if _contains_normalized_phrase(topic_normalized, section.title):
                score_by_section[section.section_id] += 4
            for alias in section.aliases:
                if _contains_normalized_phrase(topic_normalized, alias):
                    score_by_section[section.section_id] += 3

    for section_id in _DEMO_TO_BOOK_SECTIONS.get(matched_demo_id, ()):
        score_by_section[section_id] = score_by_section.get(section_id, 0) + 8

    ranked = sorted(
        (section for section in _BOOK_SECTIONS if score_by_section.get(section.section_id, 0) > 0),
        key=lambda section: (-score_by_section[section.section_id], section.start_page, section.title),
    )
    if not ranked:
        return (), ""

    selected = tuple(ranked[:limit])
    titles = ", ".join(section.title for section in selected[:2])
    if matched_demo_id:
        rationale = f"These sections line up with the matched demo and the concepts in your query, especially {titles}."
    elif topic_title:
        rationale = f"These sections are the closest book entry points for the topic {topic_title.lower()}, especially {titles}."
    else:
        rationale = f"These sections best match the concepts named in your query, especially {titles}."
    return selected, rationale


def recommend_course_project_ideas(
    query: str,
) -> tuple[str, tuple[CourseProjectIdea, ...], CourseDemoSpec | None, tuple[BookSectionSpec, ...], str]:
    normalized = _normalize_query_text(query)
    best_guide: CourseProjectGuide | None = None
    best_score = 0
    for guide in _COURSE_PROJECT_GUIDES:
        score = 0
        for alias in guide.aliases:
            if _contains_normalized_phrase(normalized, alias):
                score += 4
        if score > best_score:
            best_guide = guide
            best_score = score
    if best_guide is None or best_score == 0:
        matched_demo = match_course_demo(query)
        if matched_demo is None:
            return "", (), None, (), ""
        guide_by_demo = next((guide for guide in _COURSE_PROJECT_GUIDES if guide.starter_demo_id == matched_demo.demo_id), None)
        if guide_by_demo is None:
            return "", (), matched_demo, recommend_book_sections(query, matched_demo_id=matched_demo.demo_id)[0], ""
        best_guide = guide_by_demo
    starter_demo = _COURSE_DEMOS_BY_ID.get(best_guide.starter_demo_id)
    book_sections = tuple(
        _BOOK_SECTIONS_BY_ID[section_id]
        for section_id in best_guide.book_section_ids
        if section_id in _BOOK_SECTIONS_BY_ID
    )
    rationale = best_guide.rationale
    return best_guide.title, best_guide.ideas, starter_demo, book_sections, rationale


def recommend_course_learning_resources(
    query: str,
) -> tuple[str, tuple[CourseDemoSpec, ...], tuple[BookSectionSpec, ...], tuple[CourseCodeSnippet, ...], str]:
    normalized = _normalize_query_text(query)
    topic_title, demos, rationale = recommend_course_demos(query)
    if not topic_title:
        project_topic, _, starter_demo, project_book_sections, project_rationale = recommend_course_project_ideas(query)
        if project_topic:
            topic_title = project_topic
            demos = (starter_demo,) if starter_demo else ()
            rationale = project_rationale
            book_sections = project_book_sections
        else:
            return "", (), (), (), ""
    else:
        matched_demo_id = demos[0].demo_id if demos else ""
        book_sections, _ = recommend_book_sections(query, matched_demo_id=matched_demo_id, topic_title=topic_title)

    topic_key = ""
    for guide in _COURSE_PROJECT_GUIDES:
        if guide.title == topic_title or any(_contains_normalized_phrase(normalized, alias) for alias in guide.aliases):
            topic_key = guide.topic_id
            break
    if not topic_key and demos:
        topic_key = next((guide.topic_id for guide in _COURSE_PROJECT_GUIDES if guide.starter_demo_id == demos[0].demo_id), "")
    snippets = tuple(snippet for snippet in _COURSE_CODE_SNIPPETS if snippet.topic_id == topic_key)
    if not snippets and topic_title == "Language Modeling":
        snippets = tuple(snippet for snippet in _COURSE_CODE_SNIPPETS if snippet.topic_id == "kan_extension_transformers")
    if not snippets:
        return topic_title, demos, book_sections, (), rationale
    return topic_title, demos, book_sections, snippets, rationale


def _normalize_query_text(value: str) -> str:
    lowered = value.lower()
    lowered = lowered.replace("gt+db", "gt db")
    lowered = lowered.replace("gt‑full", "gt full")
    lowered = lowered.replace("gt-full", "gt full")
    lowered = lowered.replace("j-stability", "j stability")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _contains_normalized_phrase(haystack: str, needle: str) -> bool:
    normalized_needle = _normalize_query_text(needle)
    if not haystack or not normalized_needle:
        return False
    return f" {normalized_needle} " in f" {haystack} "


class CourseDemoAgenticRunner:
    """Choose and optionally execute one course demo notebook."""

    def __init__(self, config: CourseDemoAgenticConfig) -> None:
        self.config = config.resolved()
        if not self.config.query:
            raise ValueError("A non-empty course demo query is required.")
        self.config.outdir.mkdir(parents=True, exist_ok=True)

    def run(self) -> CourseDemoRunResult:
        learning_topic, learning_demos, learning_book_sections, learning_snippets, learning_rationale = (
            recommend_course_learning_resources(self.config.query)
        )
        learning_snippets = self._resolve_snippet_sources(learning_snippets)
        if is_course_learning_query(self.config.query) and learning_topic:
            dashboard_path = self.config.outdir / "course_demo_dashboard.html"
            summary_path = self.config.outdir / "course_demo_summary.json"
            stdout_path = self.config.outdir / "course_demo_stdout.txt"
            stderr_path = self.config.outdir / "course_demo_stderr.txt"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            selected_demo = learning_demos[0] if learning_demos else None
            notebook_path = (
                self.config.course_repo_root / selected_demo.notebook_relpath
                if selected_demo is not None and self.config.course_repo_root is not None
                else None
            )
            result = CourseDemoRunResult(
                query_plan=CourseDemoQueryPlan(
                    query=self.config.query,
                    normalized_query=" ".join(self.config.query.lower().split()),
                    explanation_focus=f"Teach the core ideas of {learning_topic.lower()} with book references, runnable demos, and code snippets.",
                    matched_demo_id=selected_demo.demo_id if selected_demo else "",
                    matched_title=selected_demo.title if selected_demo else learning_topic,
                    notebook_path=notebook_path,
                ),
                selected_demo=selected_demo,
                route_outdir=self.config.outdir,
                notebook_path=notebook_path,
                generated_script_path=None,
                dashboard_path=dashboard_path,
                summary_path=summary_path,
                response_mode="learning_guide",
                execution_attempted=False,
                execution_status="recommended",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                implementation_language="python",
                selected_demo_source_href=self._materialize_source_view(
                    notebook_path,
                    source_id=f"{selected_demo.demo_id}_starter_demo" if selected_demo else "starter_demo",
                    title=selected_demo.title if selected_demo else learning_topic,
                ),
                book_pdf_path=self.config.book_pdf_path,
                book_recommendations=learning_book_sections,
                book_rationale=(
                    f"These sections are the best reading path for learning {learning_topic.lower()}."
                    if learning_book_sections
                    else ""
                ),
                code_snippets=learning_snippets,
                recommendation_demos=learning_demos,
                recommendation_topic=learning_topic,
                error_message=learning_rationale,
            )
            summary_path.write_text(_result_summary_json(result), encoding="utf-8")
            dashboard_path.write_text(_render_dashboard_html(result), encoding="utf-8")
            return result

        project_topic, project_ideas, starter_demo, project_book_sections, project_rationale = recommend_course_project_ideas(
            self.config.query
        )
        if is_course_project_query(self.config.query) and project_topic:
            dashboard_path = self.config.outdir / "course_demo_dashboard.html"
            summary_path = self.config.outdir / "course_demo_summary.json"
            stdout_path = self.config.outdir / "course_demo_stdout.txt"
            stderr_path = self.config.outdir / "course_demo_stderr.txt"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            notebook_path = (
                self.config.course_repo_root / starter_demo.notebook_relpath
                if starter_demo is not None and self.config.course_repo_root is not None
                else None
            )
            result = CourseDemoRunResult(
                query_plan=CourseDemoQueryPlan(
                    query=self.config.query,
                    normalized_query=" ".join(self.config.query.lower().split()),
                    explanation_focus=f"Suggest scoped course projects for {project_topic.lower()}.",
                    matched_demo_id=starter_demo.demo_id if starter_demo else "",
                    matched_title=starter_demo.title if starter_demo else project_topic,
                    notebook_path=notebook_path,
                ),
                selected_demo=starter_demo,
                route_outdir=self.config.outdir,
                notebook_path=notebook_path,
                generated_script_path=None,
                dashboard_path=dashboard_path,
                summary_path=summary_path,
                response_mode="project_ideas",
                execution_attempted=False,
                execution_status="recommended",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                implementation_language="python",
                selected_demo_source_href=self._materialize_source_view(
                    notebook_path,
                    source_id=f"{starter_demo.demo_id}_starter_demo" if starter_demo else "starter_demo",
                    title=starter_demo.title if starter_demo else project_topic,
                ),
                book_pdf_path=self.config.book_pdf_path,
                book_recommendations=project_book_sections,
                book_rationale=(
                    f"These sections are the best reading path before starting a project on {project_topic.lower()}."
                    if project_book_sections
                    else ""
                ),
                project_ideas=project_ideas,
                project_topic=project_topic,
                project_rationale=project_rationale,
                recommendation_demos=(starter_demo,) if starter_demo else (),
                recommendation_topic=project_topic,
                error_message=project_rationale,
            )
            summary_path.write_text(_result_summary_json(result), encoding="utf-8")
            dashboard_path.write_text(_render_dashboard_html(result), encoding="utf-8")
            return result

        topic_title, recommendation_demos, topic_rationale = recommend_course_demos(self.config.query)
        julia_topic_title, recommendation_julia_demos, julia_topic_rationale = recommend_julia_demos(self.config.query)
        if is_course_recommendation_query(self.config.query) and (recommendation_demos or recommendation_julia_demos):
            dashboard_path = self.config.outdir / "course_demo_dashboard.html"
            summary_path = self.config.outdir / "course_demo_summary.json"
            stdout_path = self.config.outdir / "course_demo_stdout.txt"
            stderr_path = self.config.outdir / "course_demo_stderr.txt"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            selected_demo = recommendation_julia_demos[0] if recommendation_julia_demos else recommendation_demos[0]
            notebook_path = None
            topic_name = julia_topic_title if recommendation_julia_demos else topic_title
            topic_reason = julia_topic_rationale if recommendation_julia_demos else topic_rationale
            implementation_language = "julia" if recommendation_julia_demos else "python"
            book_recommendations, book_rationale = recommend_book_sections(
                self.config.query,
                matched_demo_id=selected_demo.demo_id,
                topic_title=topic_name,
            )
            if recommendation_demos:
                notebook_path = (
                    self.config.course_repo_root / recommendation_demos[0].notebook_relpath if self.config.course_repo_root else None
                )
            result = CourseDemoRunResult(
                query_plan=CourseDemoQueryPlan(
                    query=self.config.query,
                    normalized_query=" ".join(self.config.query.lower().split()),
                    explanation_focus=f"Recommend the best starting demos for {topic_name.lower()}.",
                    matched_demo_id=selected_demo.demo_id,
                    matched_title=selected_demo.title,
                    notebook_path=notebook_path,
                ),
                selected_demo=selected_demo,
                route_outdir=self.config.outdir,
                notebook_path=notebook_path,
                generated_script_path=None,
                dashboard_path=dashboard_path,
                summary_path=summary_path,
                response_mode="recommendation",
                execution_attempted=False,
                execution_status="recommended",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                implementation_language=implementation_language,
                selected_demo_source_href="",
                book_pdf_path=self.config.book_pdf_path,
                book_recommendations=book_recommendations,
                book_rationale=book_rationale,
                recommendation_demos=recommendation_demos,
                recommendation_julia_demos=recommendation_julia_demos,
                recommendation_topic=topic_name,
                error_message=topic_reason,
            )
            summary_path.write_text(_result_summary_json(result), encoding="utf-8")
            dashboard_path.write_text(_render_dashboard_html(result), encoding="utf-8")
            return result

        julia_demo = match_julia_demo(self.config.query)
        if julia_demo is not None:
            return self._run_julia_demo(julia_demo)

        selected_demo = match_course_demo(self.config.query)
        if selected_demo is None:
            raise ValueError(
                "CLIFF could not match that request to a registered Category Theory for AGI demo."
            )
        repo_root = self.config.course_repo_root
        if repo_root is None or not repo_root.exists():
            raise FileNotFoundError(
                "The Category-Theory-for-AGI-UMass-CMPSCI-692CT repo was not found. "
                "Pass --course-repo-root to point CLIFF at the course workspace."
            )

        notebook_path = repo_root / selected_demo.notebook_relpath
        if not notebook_path.exists():
            raise FileNotFoundError(f"Course demo notebook not found: {notebook_path}")

        plan = CourseDemoQueryPlan(
            query=self.config.query,
            normalized_query=" ".join(self.config.query.lower().split()),
            explanation_focus=_explanation_focus_for_query(self.config.query, selected_demo),
            matched_demo_id=selected_demo.demo_id,
            matched_title=selected_demo.title,
            notebook_path=notebook_path,
        )
        book_recommendations, book_rationale = recommend_book_sections(
            self.config.query,
            matched_demo_id=selected_demo.demo_id,
        )
        generated_script_path = self.config.outdir / f"{selected_demo.demo_id}_extracted.py"
        stdout_path = self.config.outdir / "course_demo_stdout.txt"
        stderr_path = self.config.outdir / "course_demo_stderr.txt"
        summary_path = self.config.outdir / "course_demo_summary.json"
        dashboard_path = self.config.outdir / "course_demo_dashboard.html"

        generated_script_path.write_text(_extract_notebook_script(notebook_path), encoding="utf-8")

        execution_attempted = self.config.execute_demo
        execution_status = "not_requested"
        execution_returncode: int | None = None
        execution_seconds: float | None = None
        error_message = ""
        if execution_attempted:
            start = time.time()
            execution_status, execution_returncode, error_message = self._execute_script(
                generated_script_path=generated_script_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            execution_seconds = round(time.time() - start, 3)
        else:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")

        result = CourseDemoRunResult(
            query_plan=plan,
            selected_demo=selected_demo,
            route_outdir=self.config.outdir,
            notebook_path=notebook_path,
            generated_script_path=generated_script_path,
            dashboard_path=dashboard_path,
            summary_path=summary_path,
            response_mode="demo_run",
            execution_attempted=execution_attempted,
            execution_status=execution_status,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            book_pdf_path=self.config.book_pdf_path,
            book_recommendations=book_recommendations,
            book_rationale=book_rationale,
            execution_returncode=execution_returncode,
            execution_seconds=execution_seconds,
            error_message=error_message,
        )
        summary_path.write_text(_result_summary_json(result), encoding="utf-8")
        dashboard_path.write_text(_render_dashboard_html(result), encoding="utf-8")
        return result

    def _run_julia_demo(self, demo: JuliaDemoSpec) -> CourseDemoRunResult:
        repo_root = self.config.julia_repo_root if demo.repo_kind == "functorflow_jl" else self.config.julia_examples_root
        if repo_root is None or not repo_root.exists():
            raise FileNotFoundError(f"The Julia demo root was not found for {demo.title}.")

        source_path = repo_root / demo.source_relpath
        if demo.execution_mode != "inline" and not source_path.exists():
            raise FileNotFoundError(f"Julia demo source not found: {source_path}")

        generated_script_path = self.config.outdir / f"{demo.demo_id}_extracted.jl"
        stdout_path = self.config.outdir / "course_demo_stdout.txt"
        stderr_path = self.config.outdir / "course_demo_stderr.txt"
        summary_path = self.config.outdir / "course_demo_summary.json"
        dashboard_path = self.config.outdir / "course_demo_dashboard.html"
        book_recommendations, book_rationale = recommend_book_sections(
            self.config.query,
            matched_demo_id=demo.demo_id,
        )

        script_path = source_path
        if demo.execution_mode == "inline":
            script_payload = demo.inline_script.replace(
                "/Users/sridharmahadevan/Documents/Playground/FunctorFlow.jl",
                str(self.config.julia_repo_root),
            )
            generated_script_path.write_text(script_payload + "\n", encoding="utf-8")
            script_path = generated_script_path

        execution_attempted = self.config.execute_demo
        execution_status = "not_requested"
        execution_returncode: int | None = None
        execution_seconds: float | None = None
        error_message = ""
        if execution_attempted:
            start = time.time()
            execution_status, execution_returncode, error_message = self._execute_julia_script(
                script_path=script_path,
                cwd=repo_root,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            execution_seconds = round(time.time() - start, 3)
        else:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")

        result = CourseDemoRunResult(
            query_plan=CourseDemoQueryPlan(
                query=self.config.query,
                normalized_query=" ".join(self.config.query.lower().split()),
                explanation_focus=f"Use the Julia implementation of {demo.title.lower()} for this query.",
                matched_demo_id=demo.demo_id,
                matched_title=demo.title,
                notebook_path=source_path,
            ),
            selected_demo=demo,
            route_outdir=self.config.outdir,
            notebook_path=source_path,
            generated_script_path=script_path,
            dashboard_path=dashboard_path,
            summary_path=summary_path,
            response_mode="demo_run",
            execution_attempted=execution_attempted,
            execution_status=execution_status,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            implementation_language="julia",
            selected_demo_source_href=self._materialize_source_view(
                source_path,
                source_id=f"{demo.demo_id}_starter_demo",
                title=demo.title,
            ),
            book_pdf_path=self.config.book_pdf_path,
            book_recommendations=book_recommendations,
            book_rationale=book_rationale,
            execution_returncode=execution_returncode,
            execution_seconds=execution_seconds,
            error_message=error_message,
        )
        summary_path.write_text(_result_summary_json(result), encoding="utf-8")
        dashboard_path.write_text(_render_dashboard_html(result), encoding="utf-8")
        return result

    def _execute_script(
        self,
        *,
        generated_script_path: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> tuple[str, int | None, str]:
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        mpl_config_dir = Path("/tmp/cliff-course-demo-mplconfig")
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        try:
            completed = subprocess.run(
                [sys.executable, str(generated_script_path)],
                cwd=str(self.config.course_repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.execution_timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text((exc.stderr or "") + "\nExecution timed out.", encoding="utf-8")
            return "timeout", None, f"Timed out after {self.config.execution_timeout_sec} seconds."

        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode == 0:
            return "completed", 0, ""
        return "failed", completed.returncode, f"Notebook-derived script exited with code {completed.returncode}."

    def _execute_julia_script(
        self,
        *,
        script_path: Path,
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> tuple[str, int | None, str]:
        env = os.environ.copy()
        env.setdefault("JULIA_PROJECT", str(self.config.julia_repo_root))
        julia_depot = _default_julia_depot_path()
        julia_depot.mkdir(parents=True, exist_ok=True)
        env.setdefault("JULIA_DEPOT_PATH", str(julia_depot))
        env.setdefault("JULIAUP_DEPOT_PATH", str(julia_depot))
        availability_error = _julia_runtime_error(env)
        if availability_error:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(availability_error + "\n", encoding="utf-8")
            return "unavailable", None, availability_error
        try:
            completed = subprocess.run(
                [_resolve_julia_bin(), "--project=" + str(self.config.julia_repo_root), str(script_path)],
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.execution_timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text((exc.stderr or "") + "\nExecution timed out.", encoding="utf-8")
            return "timeout", None, f"Timed out after {self.config.execution_timeout_sec} seconds."

        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode == 0:
            return "completed", 0, ""
        return "failed", completed.returncode, f"Julia demo exited with code {completed.returncode}."

    def _resolve_snippet_sources(
        self,
        snippets: tuple[CourseCodeSnippet, ...],
    ) -> tuple[CourseCodeSnippet, ...]:
        resolved: list[CourseCodeSnippet] = []
        for snippet in snippets:
            source_path: Path | None = None
            if snippet.repo_kind == "course_repo" and self.config.course_repo_root is not None:
                source_path = self.config.course_repo_root / snippet.source_relpath
            elif snippet.repo_kind == "functorflow_jl" and self.config.julia_repo_root is not None:
                source_path = self.config.julia_repo_root / snippet.source_relpath
            elif snippet.repo_kind == "julia_examples" and self.config.julia_examples_root is not None:
                source_path = self.config.julia_examples_root / snippet.source_relpath
            source_href = self._materialize_source_view(
                source_path,
                source_id=snippet.snippet_id,
                title=snippet.title,
            )
            resolved.append(replace(snippet, source_href=source_href))
        return tuple(resolved)

    def _materialize_source_view(
        self,
        source_path: Path | None,
        *,
        source_id: str,
        title: str,
    ) -> str:
        if source_path is None or not source_path.exists():
            return ""
        source_views_dir = self.config.outdir / "source_views"
        source_views_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", source_id).strip("_") or "source"
        target_path = source_views_dir / f"{safe_id}.html"
        escaped_title = html.escape(title)
        escaped_label = html.escape(str(source_path))
        escaped_body = html.escape(source_path.read_text(encoding="utf-8"))
        target_path.write_text(
            f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escaped_title}</title>
    <style>
      body {{ margin: 0; font-family: Georgia, "Iowan Old Style", serif; background: #f5f1e8; color: #17231d; }}
      main {{ max-width: 980px; margin: 32px auto; padding: 0 18px 40px; }}
      .card {{ background: rgba(255, 252, 246, 0.96); border: 1px solid #d5c8af; border-radius: 24px; padding: 24px; }}
      .eyebrow {{ margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; color: #205c48; }}
      p {{ color: #536258; line-height: 1.6; font-size: 16px; }}
      pre {{ margin: 14px 0 0 0; padding: 16px; border-radius: 18px; border: 1px solid #d5c8af; background: #f8f4ec; white-space: pre-wrap; word-break: break-word; overflow-x: auto; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <p class="eyebrow">CLIFF Source View</p>
        <h1>{escaped_title}</h1>
        <p><span class="mono">{escaped_label}</span></p>
        <pre>{escaped_body}</pre>
      </section>
    </main>
  </body>
</html>
""",
            encoding="utf-8",
        )
        return target_path.relative_to(self.config.outdir).as_posix()


def _julia_runtime_error(env: dict[str, str]) -> str:
    try:
        completed = subprocess.run(
            [_resolve_juliaup_bin(), "status"],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return f"CLIFF could not verify the Julia runtime: {exc}"
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        return stderr or "CLIFF could not query juliaup for an installed Julia runtime."
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) <= 2:
        return (
            "CLIFF found juliaup but no installed Julia channel is available in this environment. "
            "Install or configure a Julia channel, then retry the Julia demo."
        )
    return ""


def _extract_notebook_script(notebook_path: Path) -> str:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    blocks: list[str] = [
        "# Auto-generated by CLIFF from a course notebook.",
        f"# Source notebook: {notebook_path}",
        "",
        "import os",
        'os.environ.setdefault("MPLBACKEND", "Agg")',
        "",
        "try:",
        "    import matplotlib",
        '    matplotlib.use("Agg", force=True)',
        "    import matplotlib.pyplot as plt",
        "    plt.show = lambda *args, **kwargs: None",
        "except Exception:",
        "    pass",
        "",
    ]
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source_lines = []
        for raw_line in cell.get("source", []):
            line = raw_line.rstrip("\n")
            stripped = line.lstrip()
            if stripped.startswith("%") or stripped.startswith("!"):
                source_lines.append(f"# Skipped notebook magic: {line}")
                continue
            source_lines.append(line)
        block = "\n".join(source_lines).strip()
        if block:
            blocks.append(block)
            blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def _result_summary_json(result: CourseDemoRunResult) -> str:
    return json.dumps(
        {
            "query_plan": {
                **asdict(result.query_plan),
                "notebook_path": str(result.query_plan.notebook_path),
            },
            "selected_demo": asdict(result.selected_demo) if result.selected_demo else None,
            "route_outdir": str(result.route_outdir),
            "notebook_path": str(result.notebook_path) if result.notebook_path else None,
            "generated_script_path": str(result.generated_script_path) if result.generated_script_path else None,
            "dashboard_path": str(result.dashboard_path),
            "response_mode": result.response_mode,
            "implementation_language": result.implementation_language,
            "book_pdf_path": str(result.book_pdf_path) if result.book_pdf_path else None,
            "book_recommendations": [asdict(section) for section in result.book_recommendations],
            "book_rationale": result.book_rationale,
            "code_snippets": [asdict(snippet) for snippet in result.code_snippets],
            "project_topic": result.project_topic,
            "project_rationale": result.project_rationale,
            "project_ideas": [asdict(idea) for idea in result.project_ideas],
            "recommendation_topic": result.recommendation_topic,
            "recommendation_demos": [asdict(demo) for demo in result.recommendation_demos],
            "recommendation_julia_demos": [asdict(demo) for demo in result.recommendation_julia_demos],
            "execution_attempted": result.execution_attempted,
            "execution_status": result.execution_status,
            "execution_returncode": result.execution_returncode,
            "execution_seconds": result.execution_seconds,
            "stdout_path": str(result.stdout_path),
            "stderr_path": str(result.stderr_path),
            "error_message": result.error_message,
        },
        indent=2,
    )


def _render_dashboard_html(result: CourseDemoRunResult) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    book_pdf_label = esc(result.book_pdf_path) if result.book_pdf_path else "catagi.pdf"
    book_suggestions = "".join(
        (
            "<li>"
            f"<strong>{esc(section.title)}</strong> "
            f"<span class=\"mono\">page {esc(section.start_page)}</span><br />"
            f"{esc(section.description)}"
            "</li>"
        )
        for section in result.book_recommendations
    )
    book_section_html = ""
    if book_suggestions:
        book_section_html = (
            "<h2>Book chapters to read first</h2>"
            f"<p>CLIFF matched your topic to sections in <span class=\"mono\">{book_pdf_label}</span>.</p>"
            f"<p>{esc(result.book_rationale)}</p>"
            f"<ul>{book_suggestions}</ul>"
        )
    project_idea_html = "".join(
        (
            "<li>"
            f"<strong>{esc(idea.title)}</strong> "
            f"<span class=\"mono\">{esc(idea.difficulty)}</span><br />"
            f"{esc(idea.summary)}<br />"
            f"<strong>Deliverables:</strong> {esc(', '.join(idea.deliverables))}"
            + (f"<br /><strong>Stretch goal:</strong> {esc(idea.stretch_goal)}" if idea.stretch_goal else "")
            + "</li>"
        )
        for idea in result.project_ideas
    )
    snippet_html = "".join(
        (
            "<li>"
            f"<strong>{esc(snippet.title)}</strong> "
            f"<span class=\"mono\">{esc(snippet.language)} · {esc(snippet.source_relpath)}</span><br />"
            f"{esc(snippet.description)}"
            + (
                f'<br /><a href="{esc(snippet.source_href)}" target="_blank" rel="noopener noreferrer">Open source</a>'
                if snippet.source_href
                else ""
            )
            + (f"<br /><strong>Try next:</strong> {esc(snippet.follow_up)}" if snippet.follow_up else "")
            + f"<pre>{esc(snippet.snippet)}</pre>"
            + "</li>"
        )
        for snippet in result.code_snippets
    )
    starter_demo_html = ""
    if result.selected_demo is not None:
        starter_links: list[str] = []
        if result.selected_demo_source_href:
            starter_links.append(
                f'<a href="{esc(result.selected_demo_source_href)}" target="_blank" rel="noopener noreferrer">Open source</a>'
            )
        if isinstance(result.selected_demo, CourseDemoSpec):
            starter_links.append(f'<a href="{esc(result.selected_demo.colab_url)}">Open in Colab</a>')
        starter_link_html = f"<br />{' · '.join(starter_links)}" if starter_links else ""
        starter_demo_html = (
            "<h2>Starter demo</h2>"
            f"<p><strong>{esc(result.selected_demo.title)}</strong><br />"
            f"{esc(result.selected_demo.description)}"
            f"{starter_link_html}</p>"
        )

    if result.response_mode == "project_ideas":
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(result.project_topic or "Course Project Ideas")}</title>
    <style>
      :root {{
        --ink: #17231d;
        --muted: #536258;
        --paper: #f5f1e8;
        --card: rgba(255, 252, 246, 0.96);
        --line: #d5c8af;
        --accent: #205c48;
      }}
      body {{ margin: 0; font-family: Georgia, "Iowan Old Style", serif; color: var(--ink); background: linear-gradient(180deg, #fcf8f0 0%, var(--paper) 100%); }}
      main {{ max-width: 980px; margin: 40px auto; padding: 0 18px 40px; }}
      .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 28px; padding: 28px; box-shadow: 0 18px 54px rgba(28, 21, 12, 0.10); }}
      .eyebrow {{ margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; color: var(--accent); }}
      p, li {{ color: var(--muted); line-height: 1.6; font-size: 16px; }}
      ul {{ padding-left: 18px; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }}
      a {{ color: var(--accent); }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <p class="eyebrow">CLIFF Project Guide</p>
        <h1>{esc(result.project_topic)}</h1>
        <p>{esc(result.project_rationale or result.error_message)}</p>
        <p><strong>Query:</strong> {esc(result.query_plan.query)}</p>
        {book_section_html}
        {starter_demo_html}
        {"<h2>Project ideas</h2><ul>" + project_idea_html + "</ul>" if project_idea_html else ""}
      </section>
    </main>
  </body>
</html>
"""

    if result.response_mode == "learning_guide":
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(result.recommendation_topic or "Course Learning Guide")}</title>
    <style>
      :root {{
        --ink: #17231d;
        --muted: #536258;
        --paper: #f5f1e8;
        --card: rgba(255, 252, 246, 0.96);
        --line: #d5c8af;
        --accent: #205c48;
      }}
      body {{ margin: 0; font-family: Georgia, "Iowan Old Style", serif; color: var(--ink); background: linear-gradient(180deg, #fcf8f0 0%, var(--paper) 100%); }}
      main {{ max-width: 980px; margin: 40px auto; padding: 0 18px 40px; }}
      .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 28px; padding: 28px; box-shadow: 0 18px 54px rgba(28, 21, 12, 0.10); }}
      .eyebrow {{ margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; color: var(--accent); }}
      p, li {{ color: var(--muted); line-height: 1.6; font-size: 16px; }}
      ul {{ padding-left: 18px; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }}
      pre {{ margin: 10px 0 0 0; padding: 16px; border-radius: 18px; border: 1px solid var(--line); background: #f8f4ec; white-space: pre-wrap; word-break: break-word; }}
      a {{ color: var(--accent); }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <p class="eyebrow">CLIFF Learning Guide</p>
        <h1>{esc(result.recommendation_topic)}</h1>
        <p>{esc(result.error_message)}</p>
        <p><strong>Query:</strong> {esc(result.query_plan.query)}</p>
        {book_section_html}
        {starter_demo_html}
        {"<h2>Recommended demos</h2><ul>" + "".join("<li><strong>" + esc(demo.title) + "</strong><br />" + esc(demo.description) + "</li>" for demo in result.recommendation_demos) + "</ul>" if result.recommendation_demos else ""}
        {"<h2>Implementation snippets</h2><ul>" + snippet_html + "</ul>" if snippet_html else ""}
      </section>
    </main>
  </body>
</html>
"""

    if result.response_mode == "recommendation":
        python_suggestions = "".join(
            (
                "<li>"
                f"<strong>{esc(demo.title)}</strong> "
                f"<span class=\"mono\">{esc(demo.notebook_relpath)}</span><br />"
                f"{esc(demo.description)}<br />"
                f"<a href=\"{esc(demo.colab_url)}\">Open in Colab</a>"
                "</li>"
            )
            for demo in result.recommendation_demos
        )
        julia_suggestions = "".join(
            (
                "<li>"
                f"<strong>{esc(demo.title)}</strong> "
                f"<span class=\"mono\">{esc(demo.source_relpath)}</span><br />"
                f"{esc(demo.description)}"
                "</li>"
            )
            for demo in result.recommendation_julia_demos
        )
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(result.recommendation_topic or "Course Demo Recommendations")}</title>
    <style>
      :root {{
        --ink: #17231d;
        --muted: #536258;
        --paper: #f5f1e8;
        --card: rgba(255, 252, 246, 0.96);
        --line: #d5c8af;
        --accent: #205c48;
      }}
      body {{ margin: 0; font-family: Georgia, "Iowan Old Style", serif; color: var(--ink); background: linear-gradient(180deg, #fcf8f0 0%, var(--paper) 100%); }}
      main {{ max-width: 980px; margin: 40px auto; padding: 0 18px 40px; }}
      .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 28px; padding: 28px; box-shadow: 0 18px 54px rgba(28, 21, 12, 0.10); }}
      .eyebrow {{ margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; color: var(--accent); }}
      p, li {{ color: var(--muted); line-height: 1.6; font-size: 16px; }}
      ul {{ padding-left: 18px; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }}
      a {{ color: var(--accent); }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <p class="eyebrow">CLIFF Topic Guide</p>
        <h1>{esc(result.recommendation_topic)}</h1>
        <p>{esc(result.error_message)}</p>
        <p><strong>Query:</strong> {esc(result.query_plan.query)}</p>
        {book_section_html}
        {"<h2>Python demos</h2><ul>" + python_suggestions + "</ul>" if python_suggestions else ""}
        {"<h2>Julia demos</h2><ul>" + julia_suggestions + "</ul>" if julia_suggestions else ""}
      </section>
    </main>
  </body>
</html>
"""

    stdout_preview = _tail_text(result.stdout_path)
    stderr_preview = _tail_text(result.stderr_path)
    status_label = {
        "completed": "Execution completed",
        "failed": "Execution failed",
        "timeout": "Execution timed out",
        "not_requested": "Execution skipped",
        "unavailable": "Runtime unavailable",
    }.get(result.execution_status, result.execution_status)
    read_first_html = ""
    if book_suggestions:
        read_first_html = (
            "<section class=\"card\">"
            "<h2>Read First in the Book</h2>"
            f"<p>CLIFF matched this request to sections in <span class=\"mono\">{book_pdf_label}</span> before running the demo.</p>"
            f"<p>{esc(result.book_rationale)}</p>"
            f"<ul>{book_suggestions}</ul>"
            "</section>"
        )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(result.selected_demo.title)}</title>
    <style>
      :root {{
        --ink: #17231d;
        --muted: #536258;
        --paper: #f5f1e8;
        --card: rgba(255, 252, 246, 0.96);
        --line: #d5c8af;
        --accent: #205c48;
        --accent-soft: #e0efe8;
        --warn: #8c3d14;
        --warn-soft: #f7e5d9;
      }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(32,92,72,0.14), transparent 26%),
          linear-gradient(180deg, #fcf8f0 0%, var(--paper) 100%);
      }}
      main {{
        max-width: 1040px;
        margin: 40px auto;
        padding: 0 18px 40px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 28px;
        box-shadow: 0 18px 54px rgba(28, 21, 12, 0.10);
        margin-bottom: 18px;
      }}
      .eyebrow {{
        margin: 0 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
        color: var(--accent);
      }}
      h1, h2 {{
        margin: 0 0 14px 0;
      }}
      p, li {{
        color: var(--muted);
        line-height: 1.6;
        font-size: 16px;
      }}
      ul {{
        margin: 0;
        padding-left: 18px;
      }}
      .status {{
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: {"var(--accent-soft)" if result.execution_status == "completed" else "var(--warn-soft)"};
        color: {"var(--accent)" if result.execution_status == "completed" else "var(--warn)"};
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}
      .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 13px;
      }}
      pre {{
        margin: 0;
        padding: 16px;
        border-radius: 18px;
        border: 1px solid var(--line);
        background: #f8f4ec;
        white-space: pre-wrap;
        word-break: break-word;
      }}
      a {{
        color: var(--accent);
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="card">
        <p class="eyebrow">CLIFF Course Demo</p>
        <h1>{esc(result.selected_demo.title)}</h1>
        <p>{esc(result.selected_demo.description)}</p>
        <p><span class="status">{esc(status_label)}</span></p>
        <ul>
          <li><strong>Query:</strong> {esc(result.query_plan.query)}</li>
          <li><strong>Focus:</strong> {esc(result.query_plan.explanation_focus)}</li>
          <li><strong>Implementation:</strong> {esc(result.implementation_language)}</li>
          <li><strong>Source:</strong> <span class="mono">{esc(result.notebook_path)}</span></li>
          <li><strong>Executed script:</strong> <span class="mono">{esc(result.generated_script_path)}</span></li>
          <li><strong>Colab:</strong> <a href="{esc(result.selected_demo.colab_url if result.selected_demo and hasattr(result.selected_demo, 'colab_url') else '')}">{esc(result.selected_demo.colab_url if result.selected_demo and hasattr(result.selected_demo, 'colab_url') else '')}</a></li>
          <li><strong>Runtime:</strong> {esc(result.execution_seconds if result.execution_seconds is not None else "n/a")} seconds</li>
          <li><strong>Error:</strong> {esc(result.error_message or "none")}</li>
        </ul>
      </section>
      {read_first_html}
      <section class="card">
        <h2>Execution Output</h2>
        <p>{esc(_execution_output_copy(result))}</p>
        <h2>Stdout</h2>
        <pre>{esc(stdout_preview or "(no stdout captured)")}</pre>
      </section>
      <section class="card">
        <h2>Stderr</h2>
        <pre>{esc(stderr_preview or "(no stderr captured)")}</pre>
      </section>
    </main>
  </body>
</html>
"""


def _tail_text(path: Path, *, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    payload = path.read_text(encoding="utf-8", errors="replace")
    if len(payload) <= max_chars:
        return payload
    return "...\n" + payload[-max_chars:]


def _execution_output_copy(result: CourseDemoRunResult) -> str:
    if result.implementation_language == "julia":
        return (
            "CLIFF ran the Julia FunctorFlow demo and captured its stdout and stderr. "
            "If the run failed, the stderr preview below usually identifies missing packages, project setup issues, or script-level assumptions."
        )
    return (
        "CLIFF extracted the notebook code into a Python script and ran it from the course repo root. "
        "If the run failed, the stderr preview below usually identifies missing packages, network downloads, or notebook-only assumptions."
    )


def _explanation_focus_for_query(query: str, demo: CourseDemoSpec) -> str:
    normalized = " ".join(query.lower().split())
    if "explain" in normalized:
        return f"Explain the mechanics of {demo.title.lower()} using the course demo output."
    if "show how" in normalized:
        return f"Show how {demo.title.lower()} behaves by running the demo and surfacing the key outputs."
    return f"Use {demo.title.lower()} as the concrete course example for this query."
