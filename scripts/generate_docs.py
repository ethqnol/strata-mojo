#!/usr/bin/env python3
"""
Automated Documentation Generator for Strata.

Parses all Mojo source files across `strata/`, extracts docstrings, signatures, and traits,
and generates dedicated Reference Markdown pages for each struct/trait/function,
along with unified search indices and the JSON data bundle for the interactive SPA.
"""

import json
import re
import shutil
import textwrap
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
STRATA_DIR = ROOT_DIR / "strata"
DOCS_DIR = ROOT_DIR / "docs"
REF_DIR = DOCS_DIR / "reference"
DOCS_BASE_URL = "https://ethqnol.github.io/strata-mojo"

try:
    from scripts.generate_sitemap import generate_sitemap
except ImportError:
    from generate_sitemap import generate_sitemap

MODULE_METADATA = {
    "core": {
        "title": "Core Matrix & Linear Algebra",
        "description": "Fundamental 2D dense Matrix, MatrixView, CSRMatrix, CSCMatrix sparse representations, Dataset containers, and hardware-accelerated BLAS/LAPACK solvers.",
        "files": [
            "core/matrix.mojo",
            "core/view.mojo",
            "core/sparse.mojo",
            "core/dataset.mojo",
            "core/linalg.mojo",
            "core/lapack.mojo",
            "core/interop.mojo",
        ]
    },
    "preprocessing": {
        "title": "Feature Preprocessing",
        "description": "StandardScaler, MinMaxScaler, RobustScaler, Normalizer, Binarizer, OneHotEncoder, OrdinalEncoder, LabelEncoder, SimpleImputer, and PolynomialFeatures with streaming SIMD statistics.",
        "files": [
            "preprocessing/scaler.mojo",
            "preprocessing/normalizer.mojo",
            "preprocessing/binarizer.mojo",
            "preprocessing/encoders.mojo",
            "preprocessing/imputer.mojo",
            "preprocessing/polynomial.mojo",
        ]
    },
    "linear_model": {
        "title": "Linear & Generalized Linear Models",
        "description": "LinearRegression via OLS SVD, Ridge Regression via L2 regularization, and LogisticRegression with L-BFGS and SGD optimizers.",
        "files": [
            "linear_model/linear_regression.mojo",
            "linear_model/ridge.mojo",
            "linear_model/lasso.mojo",
            "linear_model/elastic_net.mojo",
            "linear_model/logistic_regression.mojo",
            "linear_model/sgd_regressor.mojo",
            "linear_model/sgd_classifier.mojo",
        ]
    },
    "tree": {
        "title": "Decision Trees",
        "description": "Fast recursive classification (Gini, Entropy, Log-Loss) and regression (MSE, Friedman MSE, MAE) trees with streaming histogram split calculations.",
        "files": [
            "tree/classifier.mojo",
            "tree/regressor.mojo",
        ]
    },
    "ensemble": {
        "title": "Ensemble Methods",
        "description": "Random Forest Regressors/Classifiers with bootstrap aggregation, and Histogram-based Gradient Tree Boosting (HistGradientBoostingRegressor & HistGradientBoostingClassifier) with fast binning and regularized leaf steps.",
        "files": [
            "ensemble/forest.mojo",
            "ensemble/hist_gradient_boosting_regressor.mojo",
            "ensemble/hist_gradient_boosting_classifier.mojo",
        ]
    },
    "naive_bayes": {
        "title": "Naive Bayes Classifiers",
        "description": "GaussianNB for continuous features with adaptive variance smoothing, and MultinomialNB for discrete count data with dense and sparse CSR matrix acceleration.",
        "files": [
            "naive_bayes/gaussian.mojo",
            "naive_bayes/multinomial.mojo",
        ]
    },
    "compose": {
        "title": "Composite Feature Pipelines",
        "description": "ColumnTransformer for routing heterogeneous feature column subsets to distinct transformers and horizontal concatenation.",
        "files": [
            "compose/column_transformer.mojo",
        ]
    },
    "cluster": {
        "title": "Clustering Algorithms",
        "description": "Geometric partitioning K-Means with k-means++ initialization, streaming MiniBatchKMeans, and non-parametric density-based DBSCAN spatial clustering.",
        "files": [
            "cluster/kmeans.mojo",
            "cluster/minibatch_kmeans.mojo",
            "cluster/dbscan.mojo",
        ]
    },
    "decomposition": {
        "title": "Matrix Decomposition & Dimensionality Reduction",
        "description": "Principal Component Analysis (PCA) via exact SVD and TruncatedSVD with dense and sparse SpMM linear projection.",
        "files": [
            "decomposition/pca.mojo",
            "decomposition/truncated_svd.mojo",
        ]
    },
    "neighbors": {
        "title": "Nearest Neighbors",
        "description": "Distance metrics (Euclidean, Manhattan, Chebyshev, Minkowski, Cosine), nearest neighbors search, and k-NN classification and regression.",
        "files": [
            "neighbors/distance.mojo",
            "neighbors/base.mojo",
            "neighbors/classification.mojo",
            "neighbors/regression.mojo",
            "neighbors/kd_tree.mojo",
        ]
    },
    "model_selection": {
        "title": "Model Selection & Validation",
        "description": "Cross-validation splitters (K-Fold, Stratified, TimeSeries, Shuffle), cross_val_score, cross_validate, and Grid/Randomized hyperparameter search.",
        "files": [
            "model_selection/split.mojo",
            "model_selection/kfold.mojo",
            "model_selection/stratified_kfold.mojo",
            "model_selection/time_series_split.mojo",
            "model_selection/shuffle_split.mojo",
            "model_selection/stratified_shuffle_split.mojo",
            "model_selection/validation.mojo",
            "model_selection/grid_search.mojo",
            "model_selection/randomized_search.mojo",
        ]
    },
    "metrics": {
        "title": "Evaluation Metrics",
        "description": "Regression metrics (MSE, RMSE, MAE, R²), classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, Log Loss, ROC AUC), and clustering metrics (Silhouette Score).",
        "files": [
            "metrics/regression.mojo",
            "metrics/classification.mojo",
            "metrics/cluster.mojo",
        ]
    },
    "base": {
        "title": "Base Traits & Pipelines",
        "description": "Unified estimator traits (Transformer, Regressor, Classifier, Clusterer) and sequential Pipeline composition wrappers.",
        "files": [
            "base/estimator.mojo",
            "base/pipeline.mojo",
        ]
    },
    "utils": {
        "title": "Utilities & Random",
        "description": "64-bit SplitMix64 PRNG with Lemire unbiased sampling, mathematical activations (softmax, sigmoid, log_sum_exp), and validation helpers.",
        "files": [
            "utils/random.mojo",
            "utils/math.mojo",
            "utils/validation.mojo",
            "exceptions/errors.mojo",
        ]
    },
    "io": {
        "title": "Model Persistence & Serialization",
        "description": "Zero-copy, endian-safe binary serialization engine (dump, load, dumps, loads) for fitted estimators, transformers, trees, and ensembles.",
        "files": [
            "io/serializer.mojo",
        ]
    }
}


def clean_docstring(doc: str) -> str:
    """Cleans indentation, unescapes double backslashes for LaTeX formulas, and trims whitespace."""
    if not doc:
        return ""
    doc = doc.replace("\\\\", "\\")
    lines = doc.strip().split("\n")
    cleaned = [line.rstrip() for line in lines]
    return "\n".join(cleaned)


def split_args_safe(args_str: str) -> list:
    """Splits an argument string by comma while respecting nested brackets [], (), {}."""
    items = []
    current = []
    depth = 0
    for char in args_str:
        if char in "([{<":
            depth += 1
            current.append(char)
        elif char in ")]}>":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
        else:
            current.append(char)
    last_item = "".join(current).strip()
    if last_item:
        items.append(last_item)
    return items



def parse_docstring_sections(doc: str):
    """Parses NumPy, Sphinx, and Google style docstrings into structured sections."""
    sections = {
        "summary": "",
        "details": "",
        "parameters": [],
        "args": [],
        "attributes": [],
        "returns": "",
        "examples": ""
    }
    if not doc:
        return sections

    lines = doc.strip().split("\n")
    current_section = "summary"
    param_name = ""
    param_type = ""
    param_desc = []
    summary_lines = []
    detail_lines = []

    def save_current_item():
        nonlocal param_name, param_type, param_desc
        if param_name and current_section in ("parameters", "args", "attributes"):
            target_list = sections[current_section]
            target_list.append({
                "name": param_name,
                "type": param_type,
                "description": " ".join(param_desc).strip()
            })
            param_name = ""
            param_type = ""
            param_desc = []

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        header_match = re.match(r"^(Parameters|Args|Arguments|Attributes|Returns|Return|Raises|Examples?)\s*:?$", line, re.IGNORECASE)
        has_underline = i + 1 < len(lines) and lines[i + 1].strip().startswith("---")

        if header_match or has_underline:
            save_current_item()

            sec_name = (header_match.group(1) if header_match else line).lower()
            if has_underline:
                i += 2
            else:
                i += 1

            if sec_name == "parameters":
                current_section = "parameters"
            elif sec_name in ("args", "arguments"):
                current_section = "args"
            elif sec_name in ("attributes",):
                current_section = "attributes"
            elif sec_name in ("returns", "return"):
                current_section = "returns"
            elif sec_name in ("examples", "example"):
                current_section = "examples"
            elif sec_name in ("raises",):
                current_section = "raises"
            continue

        if current_section == "summary":
            if not line and summary_lines:
                current_section = "details"
            elif line:
                summary_lines.append(line)
        elif current_section == "details":
            if line:
                detail_lines.append(line)
        elif current_section in ("parameters", "args", "attributes"):
            param_match = re.match(r"^([a-zA-Z0-9_]+)(?:\s*\(([^)]+)\))?\s*:\s*(.*)$", line)
            if param_match:
                save_current_item()
                param_name = param_match.group(1)
                explicit_type = param_match.group(2)
                rest = param_match.group(3).strip()

                if explicit_type:
                    param_type = explicit_type.strip()
                    param_desc = [rest] if rest else []
                else:
                    is_type_token = re.match(r"^[A-Za-z0-9_]+(?:\[[A-Za-z0-9_, ]+\])?$", rest)
                    if is_type_token and rest not in ("Whether", "The", "If", "A", "An", "Default", "Number", "Fraction", "Proportion", "Strategy", "Impurity", "Split", "Solver", "Regularization", "Maximum", "Minimum"):
                        param_type = rest
                        param_desc = []
                    else:
                        param_type = ""
                        param_desc = [rest] if rest else []
            elif param_name and line:
                param_desc.append(line)

        elif current_section == "returns":
            if line:
                sections["returns"] += (" " if sections["returns"] else "") + line
        elif current_section == "examples":
            sections["examples"] += "\n" + raw_line

        i += 1

    save_current_item()

    sections["summary"] = " ".join(summary_lines).strip()
    sections["details"] = "\n".join(detail_lines).strip()
    sections["returns"] = sections["returns"].strip()
    sections["examples"] = sections["examples"].strip()
    return sections


def extract_mojo_symbols(filepath: Path):
    """Extracts structs, traits, functions, methods, and docstrings from a Mojo source file."""
    if not filepath.exists():
        return []

    content = filepath.read_text(encoding="utf-8")
    symbols = []

    type_pattern = re.compile(
        r'^(struct|trait)\s+([A-Za-z0-9_]+)(?:\[([\s\S]*?)\])?(?:\s*\(([\s\S]*?)\))?\s*:(?:\s*\n\s*"""([\s\S]*?)""")?',
        re.MULTILINE
    )

    fn_pattern = re.compile(
        r'(?:def|fn)\s+([A-Za-z0-9_]+)(?:\[([^\]]*)\])?\s*\((.*?)\)(?:\s*(?:raises\s*)?(?:->\s*([^{:]+?))?)?(?:\s*raises)?\s*:\s*(?:"""(.*?)""")?',
        re.DOTALL
    )

    type_spans = []

    for match in type_pattern.finditer(content):
        kind = match.group(1) # "struct" or "trait"
        name = match.group(2)
        raw_params = match.group(3) or ""
        raw_traits = match.group(4) or ""
        doc = match.group(5) or ""

        params_clean = " ".join([p.strip() for p in raw_params.split("\n") if p.strip()]).rstrip(", ")
        traits_clean = " ".join([t.strip() for t in raw_traits.split("\n") if t.strip()]).rstrip(", ")
        parsed_doc = parse_docstring_sections(clean_docstring(doc))

        type_start = match.end()
        next_match = type_pattern.search(content, type_start)
        type_end = next_match.start() if next_match else len(content)
        type_spans.append((match.start(), type_end))
        type_body = content[type_start:type_end]

        constructors = []
        methods = []
        for fn_match in fn_pattern.finditer(type_body):
            fn_name = fn_match.group(1)
            raw_fn_params = fn_match.group(2) or ""
            raw_fn_args = fn_match.group(3) or ""
            raw_fn_ret = fn_match.group(4) or ""
            fn_doc = fn_match.group(5) or ""

            if fn_name == "__init__":
                if "copy: Self" in raw_fn_args:
                    continue
                fn_params_clean = " ".join([p.strip() for p in raw_fn_params.split("\n") if p.strip()]).rstrip(", ")
                fn_args_clean = " ".join([a.strip() for a in raw_fn_args.split("\n") if a.strip()]).rstrip(", ")
                fn_parsed_doc = parse_docstring_sections(clean_docstring(fn_doc))
                constructors.append({
                    "name": "__init__",
                    "type_params": fn_params_clean,
                    "args": fn_args_clean,
                    "docstring": clean_docstring(fn_doc),
                    "parsed_doc": fn_parsed_doc,
                })
                continue

            if fn_name.startswith("_"):
                continue

            fn_params_clean = " ".join([p.strip() for p in raw_fn_params.split("\n") if p.strip()]).rstrip(", ")
            fn_args_clean = " ".join([a.strip() for a in raw_fn_args.split("\n") if a.strip()]).rstrip(", ")
            fn_ret_clean = " ".join([r.strip() for r in raw_fn_ret.split("\n") if r.strip()]).replace("raises", "").strip()
            fn_parsed_doc = parse_docstring_sections(clean_docstring(fn_doc))

            methods.append({
                "name": fn_name,
                "type_params": fn_params_clean,
                "args": fn_args_clean,
                "returns": fn_ret_clean,
                "docstring": clean_docstring(fn_doc),
                "parsed_doc": fn_parsed_doc
            })

        # Enrich with polymorphic Dataset trait overloads
        trait_tokens = [t.strip() for t in traits_clean.split(",") if t.strip()]
        existing_signatures = set((m["name"], "dataset" in m["args"]) for m in methods)

        if "Classifier" in trait_tokens:
            if ("fit", True) not in existing_signatures:
                methods.append({
                    "name": "fit",
                    "type_params": "feat_dtype: DType, target_dtype: DType",
                    "args": "mut self, dataset: Dataset[feat_dtype, target_dtype]",
                    "returns": "",
                    "docstring": "Fits the classifier using a unified Dataset container.",
                    "parsed_doc": {"summary": "Fits the classifier using a unified Dataset container.", "parameters": [{"name": "dataset", "type": "Dataset[feat_dtype, target_dtype]", "description": "**[Overloaded Alternative: Dataset Input]** Unified dataset container holding feature matrix and target labels. (Can be supplied alternatively in place of (X, y))"}], "returns": "", "details": "", "attributes": [], "examples": ""}
                })
            if ("predict", True) not in existing_signatures:
                methods.append({
                    "name": "predict",
                    "type_params": "feat_dtype: DType, target_dtype: DType",
                    "args": "self, dataset: Dataset[feat_dtype, target_dtype]",
                    "returns": "List[Int]",
                    "docstring": "Predicts class labels for a Dataset container.",
                    "parsed_doc": {"summary": "Predicts class labels for a Dataset container.", "parameters": [{"name": "dataset", "type": "Dataset[feat_dtype, target_dtype]", "description": "**[Overloaded Alternative: Dataset Input]** Unified dataset container holding feature records."}], "returns": "List[Int] of discrete predicted class labels.", "details": "", "attributes": [], "examples": ""}
                })
            if ("predict_proba", True) not in existing_signatures:
                methods.append({
                    "name": "predict_proba",
                    "type_params": "feat_dtype: DType, target_dtype: DType",
                    "args": "self, dataset: Dataset[feat_dtype, target_dtype]",
                    "returns": "Matrix[feat_dtype]",
                    "docstring": "Predicts class probability distributions for a Dataset container.",
                    "parsed_doc": {"summary": "Predicts class probability distributions for a Dataset container.", "parameters": [{"name": "dataset", "type": "Dataset[feat_dtype, target_dtype]", "description": "**[Overloaded Alternative: Dataset Input]** Unified dataset container holding feature records."}], "returns": "Matrix[feat_dtype] of predicted class probabilities.", "details": "", "attributes": [], "examples": ""}
                })

        elif "Regressor" in trait_tokens:
            if ("fit", True) not in existing_signatures:
                methods.append({
                    "name": "fit",
                    "type_params": "feat_dtype: DType, target_dtype: DType",
                    "args": "mut self, dataset: Dataset[feat_dtype, target_dtype]",
                    "returns": "",
                    "docstring": "Fits the regressor using a unified Dataset container.",
                    "parsed_doc": {"summary": "Fits the regressor using a unified Dataset container.", "parameters": [{"name": "dataset", "type": "Dataset[feat_dtype, target_dtype]", "description": "**[Overloaded Alternative: Dataset Input]** Unified dataset container holding feature matrix and targets. (Can be supplied alternatively in place of (X, y))"}], "returns": "", "details": "", "attributes": [], "examples": ""}
                })
            if ("predict", True) not in existing_signatures:
                methods.append({
                    "name": "predict",
                    "type_params": "feat_dtype: DType, target_dtype: DType",
                    "args": "self, dataset: Dataset[feat_dtype, target_dtype]",
                    "returns": "List[Scalar[feat_dtype]]",
                    "docstring": "Predicts continuous targets for a Dataset container.",
                    "parsed_doc": {"summary": "Predicts continuous targets for a Dataset container.", "parameters": [{"name": "dataset", "type": "Dataset[feat_dtype, target_dtype]", "description": "**[Overloaded Alternative: Dataset Input]** Unified dataset container holding feature records."}], "returns": "List[Scalar[feat_dtype]] of predicted values.", "details": "", "attributes": [], "examples": ""}
                })

        elif "Transformer" in trait_tokens:
            if ("fit", True) not in existing_signatures:
                methods.append({
                    "name": "fit",
                    "type_params": "feat_dtype: DType, target_dtype: DType",
                    "args": "mut self, dataset: Dataset[feat_dtype, target_dtype]",
                    "returns": "",
                    "docstring": "Fits the transformer on Dataset feature records.",
                    "parsed_doc": {"summary": "Fits the transformer on Dataset feature records.", "parameters": [{"name": "dataset", "type": "Dataset[feat_dtype, target_dtype]", "description": "**[Overloaded Alternative: Dataset Input]** Unified dataset container holding feature records. (Can be supplied alternatively in place of X)"}], "returns": "", "details": "", "attributes": [], "examples": ""}
                })
            if ("transform", True) not in existing_signatures:
                methods.append({
                    "name": "transform",
                    "type_params": "feat_dtype: DType, target_dtype: DType",
                    "args": "self, dataset: Dataset[feat_dtype, target_dtype]",
                    "returns": "Dataset[feat_dtype, target_dtype]",
                    "docstring": "Transforms dataset records, preserving column names and target metadata.",
                    "parsed_doc": {"summary": "Transforms dataset records, preserving column names and target metadata.", "parameters": [{"name": "dataset", "type": "Dataset[feat_dtype, target_dtype]", "description": "**[Overloaded Alternative: Dataset Input]** Unified dataset container to transform."}], "returns": "Dataset[feat_dtype, target_dtype] with transformed feature records.", "details": "", "attributes": [], "examples": ""}
                })
            if ("fit_transform", True) not in existing_signatures:
                methods.append({
                    "name": "fit_transform",
                    "type_params": "feat_dtype: DType, target_dtype: DType",
                    "args": "mut self, dataset: Dataset[feat_dtype, target_dtype]",
                    "returns": "Dataset[feat_dtype, target_dtype]",
                    "docstring": "Fits transformer and transforms dataset records.",
                    "parsed_doc": {"summary": "Fits transformer and transforms dataset records.", "parameters": [{"name": "dataset", "type": "Dataset[feat_dtype, target_dtype]", "description": "**[Overloaded Alternative: Dataset Input]** Unified dataset container to fit and transform."}], "returns": "Dataset[feat_dtype, target_dtype] with transformed feature records.", "details": "", "attributes": [], "examples": ""}
                })

        elif "Clusterer" in trait_tokens:
            if ("fit", True) not in existing_signatures:
                methods.append({
                    "name": "fit",
                    "type_params": "feat_dtype: DType, target_dtype: DType",
                    "args": "mut self, dataset: Dataset[feat_dtype, target_dtype]",
                    "returns": "",
                    "docstring": "Fits cluster centroids on Dataset feature records.",
                    "parsed_doc": {"summary": "Fits cluster centroids on Dataset feature records.", "parameters": [{"name": "dataset", "type": "Dataset[feat_dtype, target_dtype]", "description": "**[Overloaded Alternative: Dataset Input]** Unified dataset container holding feature records."}], "returns": "", "details": "", "attributes": [], "examples": ""}
                })
            if ("predict", True) not in existing_signatures:
                methods.append({
                    "name": "predict",
                    "type_params": "feat_dtype: DType, target_dtype: DType",
                    "args": "self, dataset: Dataset[feat_dtype, target_dtype]",
                    "returns": "List[Int]",
                    "docstring": "Predicts closest cluster assignments for a Dataset container.",
                    "parsed_doc": {"summary": "Predicts closest cluster assignments for a Dataset container.", "parameters": [{"name": "dataset", "type": "Dataset[feat_dtype, target_dtype]", "description": "**[Overloaded Alternative: Dataset Input]** Unified dataset container holding feature records."}], "returns": "List[Int] of predicted cluster indices.", "details": "", "attributes": [], "examples": ""}
                })

        symbols.append({
            "kind": kind,
            "name": name,
            "type_params": params_clean,
            "traits": traits_clean,
            "docstring": clean_docstring(doc),
            "parsed_doc": parsed_doc,
            "constructors": constructors,
            "methods": methods,
            "file": str(filepath.relative_to(ROOT_DIR))
        })


    # Global standalone functions (strictly outside any struct/trait body)
    top_fn_pattern = re.compile(
        r'^(?:def|fn)\s+([A-Za-z0-9_]+)(?:\[([^\]]*)\])?\s*\((.*?)\)(?:\s*(?:raises\s*)?->\s*([^{:\n]+))?(?:\s*raises)?\s*:\s*(?:"""(.*?)""")?',
        re.MULTILINE | re.DOTALL
    )

    for fn_match in top_fn_pattern.finditer(content):
        fn_pos = fn_match.start()
        if any(start <= fn_pos < end for start, end in type_spans):
            continue

        fn_name = fn_match.group(1)
        if fn_name.startswith("_"):
            continue

        raw_fn_params = fn_match.group(2) or ""
        raw_fn_args = fn_match.group(3) or ""
        fn_ret = fn_match.group(4) or ""
        fn_doc = fn_match.group(5) or ""

        fn_params_clean = " ".join([p.strip() for p in raw_fn_params.split("\n") if p.strip()]).rstrip(", ")
        fn_args_clean = " ".join([a.strip() for a in raw_fn_args.split("\n") if a.strip()]).rstrip(", ")
        fn_parsed_doc = parse_docstring_sections(clean_docstring(fn_doc))

        symbols.append({
            "kind": "function",
            "name": fn_name,
            "type_params": fn_params_clean,
            "args": fn_args_clean,
            "returns": fn_ret.strip(),
            "docstring": clean_docstring(fn_doc),
            "parsed_doc": fn_parsed_doc,
            "file": str(filepath.relative_to(ROOT_DIR))
        })

    return symbols


def format_examples_markdown(examples: str) -> list:
    """Formats docstring examples into clean markdown code blocks without nested fences."""
    if not examples or not examples.strip():
        return []

    lines = [l for l in textwrap.dedent(examples).strip().split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return []

    if lines[0].strip().startswith("```"):
        lines.pop(0)
    if lines and lines[-1].strip().startswith("```"):
        lines.pop()

    cleaned_code = textwrap.dedent("\n".join(lines)).strip()
    if not cleaned_code:
        return []

    return [
        "## Example",
        "",
        "```mojo",
        cleaned_code,
        "```",
        ""
    ]


def generate_symbol_markdown_page(mod_key: str, s: dict) -> str:
    """Generates a dedicated Reference Markdown page for a single struct, trait, or function."""
    name = s["name"]
    kind = s["kind"]


    pdoc = s.get("parsed_doc", {})
    file_path = s.get("file", "")

    md = []
    md.append(f"# `{name}`")
    md.append("")

    traits_str = f" &bull; **Traits**: `{s['traits']}`" if s.get("traits") else ""
    md.append(f"**Module**: [`strata.{mod_key}`](index.md) &bull; **Kind**: `{kind}`{traits_str}  ")
    md.append(f"**Source**: [`{file_path}`](file:///{ROOT_DIR}/{file_path})")
    md.append("")

    if kind in ("struct", "trait"):
        type_params = f"[{s['type_params']}]" if s["type_params"] else ""
        traits_decl = f"({s['traits']})" if s.get("traits") else ""

        # Declaration Box
        md.append("```mojo")
        md.append(f"{kind} {name}{type_params}{traits_decl}")
        md.append("```")
        md.append("")

        # Quick Import Box
        md.append("```mojo")
        md.append(f"from strata.{mod_key} import {name}")
        md.append("```")
        md.append("")

        # Summary & Overview
        if pdoc.get("summary"):
            md.append(f"**{pdoc['summary']}**")
            md.append("")
        if pdoc.get("details"):
            md.append(pdoc["details"])
            md.append("")

        # Parameters Table (Compile-Time Parameters)
        params = pdoc.get("parameters", [])
        if params:
            md.append("---")
            md.append("")
            md.append("## Parameters (Compile-Time)")
            md.append("")
            has_types = any(p["type"] for p in params)
            if has_types:
                md.append("| Parameter | Type | Description |")
                md.append("| :--- | :--- | :--- |")
                for p in params:
                    p_name = p["name"]
                    p_type = f"`{p['type']}`" if p["type"] else "—"
                    p_desc = p["description"] or "—"
                    md.append(f"| **`{p_name}`** | {p_type} | {p_desc} |")
            else:
                md.append("| Parameter | Description |")
                md.append("| :--- | :--- |")
                for p in params:
                    p_name = p["name"]
                    p_desc = p["description"] or "—"
                    md.append(f"| **`{p_name}`** | {p_desc} |")
            md.append("")

        # Constructors Section
        constructors = s.get("constructors", [])
        if constructors:
            md.append("---")
            md.append("")
            md.append("## Constructors")
            md.append("")
            for c in constructors:
                c_tparams = f"[{c['type_params']}]" if c.get("type_params") else ""
                c_args = c["args"]
                md.append("```mojo")
                md.append(f"def __init__{c_tparams}({c_args})")
                md.append("```")
                md.append("")
                c_pdoc = c.get("parsed_doc", {})
                c_sum = c_pdoc.get("summary") or c.get("docstring", "")
                if c_sum:
                    md.append(f"{c_sum}")
                    md.append("")

                sig_types = {}
                for arg in split_args_safe(c_args):
                    if arg not in ("self", "mut self", "out self", "var self"):
                        m_arg = re.match(r"^([a-zA-Z0-9_]+)\s*:\s*(.+)$", arg)
                        if m_arg:
                            sig_types[m_arg.group(1)] = m_arg.group(2).split("=")[0].strip()

                c_params = c_pdoc.get("parameters", []) or c_pdoc.get("args", [])
                if not c_params and sig_types:
                    for s_name, s_type in sig_types.items():
                        c_params.append({"name": s_name, "type": s_type, "description": "—"})

                if c_params:
                    md.append("| Argument | Type | Description |")
                    md.append("| :--- | :--- | :--- |")
                    for cp in c_params:
                        cp_name = cp["name"]
                        inferred_type = cp.get("type") or sig_types.get(cp_name, "")
                        cp_type = f"`{inferred_type}`" if inferred_type else "—"
                        cp_desc = cp.get("description", "") or "—"
                        md.append(f"| **`{cp_name}`** | {cp_type} | {cp_desc} |")
                    md.append("")

        # Arguments Table (Runtime Struct / Hyperparameter Arguments)
        args_list = pdoc.get("args", [])
        if not args_list and s.get("constructors"):
            c_first = s["constructors"][0]
            if c_first.get("parsed_doc", {}).get("args"):
                args_list = c_first["parsed_doc"]["args"]
            elif c_first.get("parsed_doc", {}).get("parameters"):
                args_list = c_first["parsed_doc"]["parameters"]

        if args_list and not constructors:
            md.append("---")
            md.append("")
            header_title = "## Arguments (Runtime)" if params else "## Arguments"
            md.append(header_title)
            md.append("")
            has_types = any(a["type"] for a in args_list)
            if has_types:
                md.append("| Argument | Type | Description |")
                md.append("| :--- | :--- | :--- |")
                for a in args_list:
                    a_name = a["name"]
                    a_type = f"`{a['type']}`" if a["type"] else "—"
                    a_desc = a["description"] or "—"
                    md.append(f"| **`{a_name}`** | {a_type} | {a_desc} |")
            else:
                md.append("| Argument | Description |")
                md.append("| :--- | :--- |")
                for a in args_list:
                    a_name = a["name"]
                    a_desc = a["description"] or "—"
                    md.append(f"| **`{a_name}`** | {a_desc} |")
            md.append("")

        # Attributes Table
        attributes = [a for a in pdoc.get("attributes", []) if not a["name"].startswith("_")]
        if attributes:
            md.append("---")
            md.append("")
            md.append("## Attributes")
            md.append("")
            has_attr_types = any(a["type"] for a in attributes)
            if has_attr_types:
                md.append("| Attribute | Type | Description |")
                md.append("| :--- | :--- | :--- |")
                for a in attributes:
                    a_name = a["name"]
                    a_type = f"`{a['type']}`" if a['type'] else "—"
                    a_desc = a["description"] or "—"
                    md.append(f"| **`{a_name}`** | {a_type} | {a_desc} |")
            else:
                md.append("| Attribute | Description |")
                md.append("| :--- | :--- |")
                for a in attributes:
                    a_name = a["name"]
                    a_desc = a["description"] or "—"
                    md.append(f"| **`{a_name}`** | {a_desc} |")
            md.append("")


        # Methods Index & Detail Cards
        if s.get("methods"):
            public_methods = [m for m in s["methods"] if not m["name"].startswith("_")]
            if public_methods:
                md.append("---")
                md.append("")
                md.append("## Methods Overview")
                md.append("")
                md.append("| Method | Description |")
                md.append("| :--- | :--- |")

                # Group methods by name for clean overload handling
                methods_by_name = {}
                for m in public_methods:
                    methods_by_name.setdefault(m["name"], []).append(m)

                for m_name, overloads in methods_by_name.items():
                    first_m = overloads[0]
                    m_pdoc = first_m.get("parsed_doc", {})
                    m_sum = m_pdoc.get("summary") or (first_m.get("docstring", "").split("\n")[0] if first_m.get("docstring") else "—")
                    md.append(f"| [`{name}.{m_name}()`](#{m_name.lower()}) | {m_sum} |")
                md.append("")

                md.append("---")
                md.append("")
                md.append("## Method Details")
                md.append("")

                for m_name, overloads in methods_by_name.items():
                    md.append(f"### `{name}.{m_name}()`")
                    md.append("")

                    # Show all overload signatures in a single code block
                    md.append("```mojo")
                    for m in overloads:
                        m_tparams = f"[{m['type_params']}]" if m["type_params"] else ""
                        m_args = m["args"]
                        m_ret = f" -> {m['returns']}" if m["returns"] else ""
                        md.append(f"def {m_name}{m_tparams}({m_args}){m_ret}")
                    md.append("```")
                    if len(overloads) > 1:
                        traits_str = s.get("traits", "")
                        if "Transformer" in traits_str:
                            md.append("> **Overload Note**: This method accepts either a standard `X: Matrix` or a unified `dataset: Dataset` container.")
                        elif "Regressor" in traits_str or "Classifier" in traits_str:
                            md.append("> **Overload Note**: This method accepts either standard `(X, y)` inputs or a unified `dataset: Dataset` container.")
                        else:
                            md.append("> **Overload Note**: This method supports multiple overloaded call signatures.")
                        md.append("")

                    # Find any docstring summary across overloads
                    summary_text = ""
                    for m in overloads:
                        m_pdoc = m.get("parsed_doc", {})
                        if m_pdoc.get("summary"):
                            summary_text = m_pdoc["summary"]
                            break
                        elif m.get("docstring"):
                            summary_text = m["docstring"].split("\n")[0]
                            break

                    if summary_text:
                        md.append(f"{summary_text}")
                        md.append("")

                    # Merge and deduplicate parameters across overloads
                    combined_params = []
                    seen_p_names = set()

                    for m in overloads:
                        m_pdoc = m.get("parsed_doc", {})
                        m_params = m_pdoc.get("parameters", [])
                        if not m_params and m["args"]:
                            raw_args = split_args_safe(m["args"])
                            for arg in raw_args:
                                if arg in ("self", "mut self", "out self", "var self"):
                                    continue
                                arg_match = re.match(r"^([a-zA-Z0-9_]+)\s*:\s*(.+)$", arg)
                                if arg_match:
                                    aname = arg_match.group(1)
                                    atype = arg_match.group(2).split("=")[0].strip()
                                    adesc = "Feature matrix." if aname == "X" else ("Target vector / class labels." if aname == "y" else ("Dataset container." if aname == "dataset" else "—"))
                                    m_params.append({"name": aname, "type": atype, "description": adesc})

                        for p in m_params:
                            p_copy = dict(p)
                            p_name = p_copy["name"]
                            traits_str = s.get("traits", "")
                            if p_name == "dataset":
                                if "Transformer" in traits_str:
                                    p_copy["description"] = "Dataset container holding feature matrix. *(Can be provided alternatively in place of X)*"
                                elif "Classifier" in traits_str or "Regressor" in traits_str:
                                    p_copy["description"] = "Dataset container holding feature matrix and targets. *(Can be provided alternatively in place of (X, y))* "
                                else:
                                    p_copy["description"] = "Dataset container holding data records. *(Overloaded alternative)*"
                            elif p_name == "X" and not p_copy["description"]:
                                p_copy["description"] = "Feature matrix of shape $(N, D)$."
                            elif p_name == "y" and not p_copy["description"]:
                                p_copy["description"] = "Target vector / class labels of length $N$."

                            if p_name not in seen_p_names:
                                seen_p_names.add(p_name)
                                combined_params.append(p_copy)

                    if combined_params:
                        md.append("| Parameter | Type | Description |")
                        md.append("| :--- | :--- | :--- |")
                        for mp in combined_params:
                            mp_name = mp["name"]
                            mp_type = f"`{mp['type']}`" if mp["type"] else "—"
                            mp_desc = mp["description"] or "—"
                            md.append(f"| **`{mp_name}`** | {mp_type} | {mp_desc} |")
                        md.append("")

                    # Returns
                    returns_text = ""
                    for m in overloads:
                        m_pdoc = m.get("parsed_doc", {})
                        if m_pdoc.get("returns"):
                            ret_type = f"`{m['returns']}` — " if m['returns'] else ""
                            returns_text = f"**Returns**: {ret_type}{m_pdoc['returns']}"
                            break
                        elif m.get("returns"):
                            returns_text = f"**Returns**: `{m['returns']}`"
                            break

                    if returns_text:
                        md.append(returns_text)
                        md.append("")

                    md.append("---")
                    md.append("")
        # Examples
        if pdoc.get("examples"):
            while md and (md[-1].strip() == "" or md[-1].strip() == "---"):
                md.pop()
            md.append("---")
            md.append("")
            md.extend(format_examples_markdown(pdoc["examples"]))


    elif kind == "function":
        type_params = f"[{s['type_params']}]" if s["type_params"] else ""
        args = s["args"]
        ret = f" -> {s['returns']}" if s["returns"] else ""

        md.append("```mojo")
        md.append(f"def {name}{type_params}({args}){ret}")
        md.append("```")
        md.append("")

        md.append("```mojo")
        md.append(f"from strata.{mod_key} import {name}")
        md.append("```")
        md.append("")

        if pdoc.get("summary"):
            md.append(f"**{pdoc['summary']}**")
            md.append("")
        if pdoc.get("details"):
            md.append(pdoc["details"])
            md.append("")

        if pdoc.get("parameters"):
            md.append("---")
            md.append("")
            md.append("## Parameters")
            md.append("")
            md.append("| Parameter | Type | Description |")
            md.append("| :--- | :--- | :--- |")
            for p in pdoc["parameters"]:
                p_name = p["name"]
                p_type = f"`{p['type']}`" if p["type"] else "—"
                p_desc = p["description"] or "—"
                md.append(f"| **`{p_name}`** | {p_type} | {p_desc} |")
            md.append("")

        if pdoc.get("returns"):
            ret_type = f"`{s['returns']}` — " if s['returns'] else ""
            md.append(f"**Returns**: {ret_type}{pdoc['returns']}")
            md.append("")

        if pdoc.get("examples"):
            while md and (md[-1].strip() == "" or md[-1].strip() == "---"):
                md.pop()
            md.append("---")
            md.append("")
            md.extend(format_examples_markdown(pdoc["examples"]))


    return "\n".join(md)




def generate_module_index_page(mod_key: str, meta: dict, symbols: list) -> str:
    """Generates the module index page listing all structs, traits, and functions in that module."""
    md = []
    md.append(f"# `strata.{mod_key}`")
    md.append("")
    md.append(meta["description"])
    md.append("")
    md.append("---")
    md.append("")


    structs = [s for s in symbols if s["kind"] == "struct"]
    traits = [s for s in symbols if s["kind"] == "trait"]
    functions = [s for s in symbols if s["kind"] == "function"]

    if structs:
        md.append("## Structs & Classes")
        md.append("")
        md.append("| Struct | Description |")
        md.append("| :--- | :--- |")
        for s in structs:
            name = s["name"]
            summary = s.get("parsed_doc", {}).get("summary", "") or s.get("docstring", "").split("\n")[0] or "—"
            md.append(f"| [`{name}`]({name}.md) | {summary} |")
        md.append("")

    if traits:
        md.append("## Traits")
        md.append("")
        md.append("| Trait | Description |")
        md.append("| :--- | :--- |")
        for s in traits:
            name = s["name"]
            summary = s.get("parsed_doc", {}).get("summary", "") or s.get("docstring", "").split("\n")[0] or "—"
            md.append(f"| [`{name}`]({name}.md) | {summary} |")
        md.append("")

    if functions:
        md.append("## Functions")
        md.append("")
        md.append("| Function | Description |")
        md.append("| :--- | :--- |")
        for s in functions:
            name = s["name"]
            summary = s.get("parsed_doc", {}).get("summary", "") or s.get("docstring", "").split("\n")[0] or "—"
            md.append(f"| [`{name}`]({name}.md) | {summary} |")
        md.append("")

    return "\n".join(md)


def build_docs_site_bundle(all_symbols_index: list, module_symbols_map: dict):
    """Bundles all markdown docs and search index into docs/data.js for GitHub Pages viewing."""
    reference_nav_groups = []

    for mod_key, meta in MODULE_METADATA.items():
        syms = module_symbols_map.get(mod_key, [])
        sym_items = []
        for s in syms:
            sym_items.append({
                "id": f"reference/{mod_key}/{s['name']}",
                "title": s["name"],
                "path": f"reference/{mod_key}/{s['name']}.md",
                "kind": s["kind"]
            })

        reference_nav_groups.append({
            "id": f"reference/{mod_key}/index",
            "title": f"strata.{mod_key}",
            "path": f"reference/{mod_key}/index.md",
            "symbols": sym_items
        })

    docs_bundle = {
        "navigation": [
            {
                "quadrant": "tutorials",
                "title": "Tutorials",
                "icon": "academic-cap",
                "description": "Learning-oriented step-by-step lessons for newcomers.",
                "items": [
                    {"id": "tutorials/quickstart", "title": "Getting Started with Strata", "path": "tutorials/quickstart.md"},
                    {"id": "tutorials/end_to_end_pipeline", "title": "End-to-End ML Pipelines", "path": "tutorials/end_to_end_pipeline.md"},
                ]
            },
            {
                "quadrant": "how_to",
                "title": "How-To Guides",
                "icon": "wrench-screwdriver",
                "description": "Problem-oriented guides and recipes for real-world tasks.",
                "items": [
                    {"id": "how_to/model_persistence", "title": "Model Persistence & Serialization", "path": "how_to/model_persistence.md"},
                    {"id": "how_to/hyperparameter_tuning", "title": "Hyperparameter Tuning", "path": "how_to/hyperparameter_tuning.md"},
                    {"id": "how_to/sparse_matrix_ops", "title": "Sparse Matrix Operations", "path": "how_to/sparse_matrix_ops.md"},
                    {"id": "how_to/out_of_bag_validation", "title": "Out-of-Bag Validation", "path": "how_to/out_of_bag_validation.md"},
                ]
            },
            {
                "quadrant": "reference",
                "title": "API Reference",
                "icon": "book-open",
                "description": "Information-oriented, auto-generated technical documentation of all modules.",
                "items": reference_nav_groups
            },
            {
                "quadrant": "explanation",
                "title": "Explanation",
                "icon": "light-bulb",
                "description": "Understanding-oriented discussions on architecture, algorithms, and Mojo internals.",
                "items": [
                    {"id": "explanation/memory_and_simd", "title": "Memory Model & SIMD", "path": "explanation/memory_and_simd.md"},
                    {"id": "explanation/estimator_traits", "title": "Estimator Traits & Polymorphism", "path": "explanation/estimator_traits.md"},
                    {"id": "explanation/tree_algorithms", "title": "Tree Splitting & Histograms", "path": "explanation/tree_algorithms.md"},
                    {"id": "explanation/benchmarks", "title": "Performance Benchmarks", "path": "explanation/benchmarks.md"},
                ]
            }
        ],
        "documents": {},
        "searchIndex": all_symbols_index
    }

    # Load all markdown files into docs_bundle["documents"]
    for section in docs_bundle["navigation"]:
        for item in section["items"]:
            # Main item
            doc_path = DOCS_DIR / item["path"]
            if doc_path.exists():
                docs_bundle["documents"][item["id"]] = doc_path.read_text(encoding="utf-8")

            # Nested symbols
            if "symbols" in item:
                for sym_item in item["symbols"]:
                    sym_doc_path = DOCS_DIR / sym_item["path"]
                    if sym_doc_path.exists():
                        docs_bundle["documents"][sym_item["id"]] = sym_doc_path.read_text(encoding="utf-8")

    bundle_js = "window.STRATA_DOCS = " + json.dumps(docs_bundle, indent=2) + ";\n"
    bundle_file = DOCS_DIR / "data.js"
    bundle_file.write_text(bundle_js, encoding="utf-8")
    print(f"  ✓ Bundled {bundle_file.relative_to(ROOT_DIR)} ({len(docs_bundle['documents'])} markdown docs)")


def main():
    print("Generating Strata Reference Documentation (Individual Symbol Pages)...")


    # Clean previous reference dir to avoid stale files
    if REF_DIR.exists():
        shutil.rmtree(REF_DIR)
    REF_DIR.mkdir(parents=True, exist_ok=True)

    all_symbols_index = []
    module_symbols_map = {}

    for mod_key, meta in MODULE_METADATA.items():
        mod_ref_dir = REF_DIR / mod_key
        mod_ref_dir.mkdir(parents=True, exist_ok=True)

        module_symbols = []
        for rel_file in meta["files"]:
            file_path = STRATA_DIR / rel_file
            symbols = extract_mojo_symbols(file_path)
            module_symbols.extend(symbols)

        module_symbols_map[mod_key] = module_symbols

        # 1. Generate Module Index Page (e.g. docs/reference/ensemble/index.md)
        index_content = generate_module_index_page(mod_key, meta, module_symbols)
        (mod_ref_dir / "index.md").write_text(index_content, encoding="utf-8")

        # 2. Generate Individual Symbol Pages (e.g. docs/reference/ensemble/RandomForestClassifier.md)
        for s in module_symbols:
            sym_page_content = generate_symbol_markdown_page(mod_key, s)
            sym_page_file = mod_ref_dir / f"{s['name']}.md"
            sym_page_file.write_text(sym_page_content, encoding="utf-8")

            all_symbols_index.append({
                "module": mod_key,
                "name": s["name"],
                "kind": s.get("kind", "struct"),
                "summary": s.get("parsed_doc", {}).get("summary", "") or s.get("docstring", ""),
                "ref_file": f"reference/{mod_key}/{s['name']}",
                "traits": s.get("traits", ""),
                "file": s.get("file", "")
            })

        print(f"  ✓ Generated docs/reference/{mod_key}/ (1 index + {len(module_symbols)} individual pages)")

    # Write search index
    search_index_file = DOCS_DIR / "search_index.json"
    search_index_file.write_text(json.dumps(all_symbols_index, indent=2), encoding="utf-8")
    print(f"  ✓ Generated {search_index_file.relative_to(ROOT_DIR)} ({len(all_symbols_index)} indexed symbols)")

    # Build client-side static docs bundle
    build_docs_site_bundle(all_symbols_index, module_symbols_map)

    # Sync static assets to docs/assets
    src_assets = ROOT_DIR / "assets"
    dst_assets = DOCS_DIR / "assets"
    if src_assets.exists():
        dst_assets.mkdir(parents=True, exist_ok=True)
        for asset in src_assets.iterdir():
            if asset.is_file():
                shutil.copy2(asset, dst_assets / asset.name)
        print(f"  ✓ Synced {src_assets.relative_to(ROOT_DIR)} to {dst_assets.relative_to(ROOT_DIR)}")

    # Generate sitemap.xml for SEO
    generate_sitemap(base_url=DOCS_BASE_URL, docs_dir=DOCS_DIR)

    print("✨ Documentation generation complete!")


if __name__ == "__main__":
    main()
