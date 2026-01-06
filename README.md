# DSS Social Network Analysis

This repository contains a complete Decision Support System (DSS) for analysing
clandestine social networks.  The system is designed to help a non‑technical
“spymaster” gain insight into the structure of a network, identify key actors,
discover latent factions, assess the robustness of clustering results, study
connectivity via the Kemeny constant, and allocate arrest teams while
minimising warnings within the network.  The implementation is based on two
project documents, one describing the high‑level objectives and another
detailing the methodological and mathematical requirements.

## Getting started

1. **Install dependencies**

   The DSS has been tested with Python 3.11.  Create and activate a virtual
   environment, then install the required packages:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit app**

   Use Streamlit to run the application.  The main entry point is
   `src/app.py`:

   ```bash
   streamlit run src/app.py
   ```

   The application is a multi‑page dashboard; pages can be selected via the
   Streamlit sidebar.  Each page implements one of the analytic tasks
   described in the project documents.

3. **Load your network**

   The DSS accepts networks in the Matrix Market (`.mtx`) format.  Use the
   “Upload & Overview” page to select your file.  The example network of
   62 nodes (originally from Lusseau *et al.*, 2003) is provided only for
   illustration; the DSS is generic and will operate on any `.mtx`
   adjacency matrix【429076285098930†L16-L25】.

## Repository structure

The repository is organised into a clear package hierarchy and includes
notebooks and tests to support exploration and development:

```
dss-social-network-analysis/
  README.md          – this documentation
  requirements.txt    – package dependencies
  .gitignore          – files ignored by Git
  data/               – placeholder for input data (no large files stored)
  notebooks/          – Jupyter notebooks for experimentation
  src/
    dss/
      config.py           – global configuration and defaults
      logging_config.py   – logger configuration
      types.py            – data classes and type aliases
      utils/              – helper functions (I/O, validation, caching, plotting)
      graph/              – graph building and statistics
      analytics/          – centralities, roles, communities, robustness,
                            Kemeny constant, arrest optimisation
      ui/                 – Streamlit state management and reusable UI components
      pages/              – one file per Streamlit page (upload, centrality, etc.)
    app.py            – entry point that registers all pages
  tests/              – unit tests for core components
```

### Notebooks

The notebooks under `notebooks/` provide a sandbox to experiment with
centrality measures, role similarity features, community detection, Kemeny
analysis and arrest optimisation without the Streamlit interface.  They may
serve as examples for how to use the underlying functions in `src/dss`.

### Tests

Unit tests ensure that key functions behave as expected.  Run the tests
with `pytest` after installing the requirements:

```bash
pytest -q
```

## Application pages

The Streamlit application consists of several pages, each addressing a
specific research question identified in the project documents:

1. **Upload & Overview:**  Parses the uploaded `.mtx` file,
   builds a graph, validates the adjacency matrix (symmetry, self loops, number
   of connected components), computes basic statistics (`N`, `|E|`, density)
   and renders a baseline network plot.

2. **Centrality:**  Computes and compares several centrality measures
   (degree, Katz, eigenvector, betweenness, closeness and PageRank).  The
   page includes controls for highlighting the top‑ and bottom‑ranked
   nodes, and supports combining measures via weighted sums or Borda
   count.

3. **Roles:**  Implements the role‑similarity approach of Cooper and
   Barahona (2010).  Nodes are represented by
   structural signatures (k‑hop degree vectors or random‑walk profiles), a
   similarity matrix is computed, and hierarchical or spectral clustering is
   applied.  The resulting role clusters are summarised using basic
   statistics, and the network is coloured according to cluster membership.

4. **Communities & Robustness:**  Provides community detection using
   Louvain, Girvan–Newman and spectral clustering, computes modularity `Q`,
   and evaluates robustness via perturbation tests (removing 5 % of edges and
   measuring changes in adjusted Rand index and modularity).  The
   page also compares community partitions with role clusters.

5. **Kemeny Analysis:**  Computes the Kemeny constant of the network via
   the fundamental matrix of a Markov chain.  Users can
   interactively remove nodes (via a selection widget) and observe how the
   Kemeny constant changes, with options to compute only on the largest
   connected component if the graph becomes disconnected.

6. **Arrest Optimisation:**  Formulates the assignment of nodes to two
   arresting departments as a balanced cut problem with capacity constraints,
   using an integer linear program to minimise cross‑department edges and a
   regret term based on communities and optionally centrality.  A heuristic
   fallback is provided when an ILP solver is unavailable.

7. **User Manual:**  A non‑technical guide for the spymaster, including
   usage instructions, explanations of each metric, recommended workflow and
   a glossary.

## Technical notes

* All heavy computations are cached using Streamlit’s caching mechanism to
  maintain responsiveness.
* Each Python module includes an `if __name__ == "__main__":` block to
  demonstrate example usage or run a small self‑test.  This makes it easy
  to execute individual modules during development.
* Logging is configured consistently via `dss/logging_config.py`.
* The code is typed where reasonable and includes docstrings explaining
  inputs and outputs.

## Troubleshooting

If you encounter issues running the ILP in the arrest optimisation page,
ensure that the pulp dependency is installed.  Alternatively, the
application will automatically fall back to a simple heuristic assignment.
For any other problems, consult the unit tests in `tests/` and the
notebooks for examples of expected behaviour.
