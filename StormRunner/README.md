# StormRunner

Search Agents

### Developer Installation
* Make sure you have configured an SSH key, if not - follow
[GitHub's official guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent),
and configure one.

* Download Python (>=3.12) from [Python's official website](https://www.python.org/downloads/).

* Clone the repository, you can directly clone the development branch. Then enter the projec'ts directory:
```shell
git clone --branch dev git@github.com:liorvi35/intro_to_ai_1.git && cd intro_to_ai_1/
```

* Configure a virtual environment for Python and install requirements:
```shell
python -m venv .venv && pip install -r requirements.txt
```

* Run the package, via:
```python
python -m SearchAgents <graph_file>
```
where instead of `<graph_file>` provide a path to a '.txt' file that represents a graph, in assignment's exact
format (see attached assignment's pdf file).

you can directly use one of the file, that are already provided via tests:
```python
python -m SearchAgents .\tests\graphs\graph_assignment_example.txt
```

(this will run the environment with the example graph that provided in the assignment)

### Documentation

* Read the existing docs that exists in the project, at `docs` and `SearchAgents/agents`.
