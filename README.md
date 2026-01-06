# DSS Social Network Analysis

This repository contains a complete Decision Support System (DSS) for analysing
clandestine social networks.  The system is designed to help a non‑technical
“spymaster” gain insight into the structure of a network, identify key actors,
discover latent factions, assess the robustness of clustering results, study
connectivity via the Kemeny constant, and allocate arrest teams while
minimising warnings within the network.  The implementation is based on two
project documents, one describing the high‑level objectives and another
detailing the methodological and mathematical requirements【429076285098930†L38-L115】【659313343491487†L155-L233】.

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
specific research question identified in the project documents【429076285098930†L38-L115】:

1. **Upload & Overview:**  Parses the uploaded `.mtx` file,
   builds a graph, validates the adjacency matrix (symmetry, self loops, number
   of connected components), computes basic statistics (`N`, `|E|`, density)
   and renders a baseline network plot.

2. **Centrality:**  Computes and compares several centrality measures
   (degree, Katz, eigenvector, betweenness, closeness and PageRank).  The
   page includes controls for highlighting the top‑ and bottom‑ranked
   nodes, supports combining measures via weighted sums or Borda
   count【659313343491487†L63-L83】, and allows you to select individual
   nodes from a sidebar list.  Selected nodes are highlighted on the
   network plot (with their identifiers shown directly on the nodes) and
   their centrality values are displayed in a separate table.

3. **Roles:**  Implements the role‑similarity approach of Cooper and
   Barahona (2010)【659313343491487†L84-L116】.  Nodes are represented by
   structural signatures (k‑hop degree vectors or random‑walk profiles), a
   similarity matrix is computed, and hierarchical or spectral clustering is
   applied.  The resulting role clusters are summarised using basic
   statistics, and the network is coloured according to cluster membership.
   A sidebar selector lets you choose specific nodes; selected nodes are
   highlighted on the plot and their role assignments and centrality
   statistics are displayed beneath the figure.

4. **Communities & Robustness:**  Provides community detection using
   Louvain, Girvan–Newman and spectral clustering, computes modularity `Q`,
   and evaluates robustness via perturbation tests (removing 5 % of edges and
   measuring changes in adjusted Rand index and modularity)【659313343491487†L123-L147】.  The
   page also compares community partitions with role clusters.  As in
   other pages, you can select nodes from a list to highlight them on the
   community‑coloured network plot and inspect their community and role
   assignments alongside their centrality measures.

5. **Kemeny Analysis:**  Computes the Kemeny constant of the network via
   the fundamental matrix of a Markov chain【429076285098930†L81-L94】.  Users can
   interactively remove nodes (via a selection widget) and observe how the
   Kemeny constant changes.  In the updated implementation the baseline
   value (with zero removals) is always plotted at the origin of the
   line chart.  As you remove nodes, points appear at 1, 2, … on the
   horizontal axis so you can immediately see the change relative to the
   baseline.  Removed nodes are clearly indicated on the network view:
   they are outlined in red **and** drawn with a different marker shape
   (an “×”) to differentiate them from the remaining nodes.  You can
   choose to compute the constant only on the largest connected component
   if the graph becomes disconnected【659313343491487†L170-L194】.  The current
   network (with node identifiers displayed on each marker) is shown
   below the selector so you can see exactly which nodes have been
   removed.  Hovering over a node reveals its identifier and whether it
   has been removed.

6. **Arrest Optimisation:**  Formulates the assignment of nodes to two
   arresting departments as a balanced cut problem with capacity constraints,
   using an integer linear program to minimise cross‑department edges and a
   regret term based on communities and optionally centrality【659313343491487†L195-L250】.  A heuristic
   fallback is provided when an ILP solver is unavailable.

7. **User Manual:**  A non‑technical guide for the spymaster, including
   usage instructions, explanations of each metric, recommended workflow and
   a glossary.

## Technical notes and troubleshooting

* **No hidden caching:**  To keep the code as simple and transparent as
  possible, this implementation deliberately avoids using Streamlit’s
  caching decorators or other hashing mechanisms.  As a result,
  expensive computations may take longer to recompute when inputs
  change, but the flow of data is easier to follow.  See
  `dss/utils/caching.py` for the simplified identity decorators.
* **Graph construction without deprecated API:**  NetworkX versions
  3.0 and above removed the `from_scipy_sparse_matrix` function.  The
  helper `dss/graph/build_graph.py` therefore constructs the graph
  manually by iterating over the non‑zero entries of the adjacency
  matrix.  This ensures compatibility with newer NetworkX releases.
* **`if __name__ == "__main__":` blocks:**  Each Python module
  contains a small self‑test or example usage in its `__main__`
  section.  This allows modules to be executed directly during
  development without using the Streamlit interface.
* **Logging:**  A central logger configuration is defined in
  `dss/logging_config.py`.  Import and use `get_logger(__name__)` in
  modules to write informative messages.
* **Typing and documentation:**  Where appropriate, functions and data
  classes are annotated with types and include docstrings describing
  their parameters and return values.

### Troubleshooting

If you encounter issues running the ILP in the arrest optimisation page,
ensure that the pulp dependency is installed.  Alternatively, the
application will automatically fall back to a simple heuristic assignment.
For any other problems, consult the unit tests in `tests/` and the
  notebooks for examples of expected behaviour.  Note that without
  caching some pages may recompute results on each run; this is normal
  given the simplified design.
