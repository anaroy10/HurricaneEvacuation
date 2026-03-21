from pathlib import Path
from networkx import Graph


class HurricaneGraphParser:
    def __init__(self, file_path: Path) -> None:
        """
        creates a new parser object

        Args:
            file_path (Path): path to a file which contains graph parameters in a specific format

        Examples:
            for file format examples see tests
        """
        self.__file_path: Path = file_path
        self.__args: dict[str, int] = {}
        self.__nodes: dict[int, list[str]] = {}
        self.__edges: dict[int, tuple[int, int, int, bool]] = {}

    def get_hurricane_graph(self) -> Graph:
        """
        parses the file and constructs the hurricane graph object

        Returns:
            undirected weighted graph, that represents the hurricane evacuation graph
        """
        # parses the ASCII file, updates data members accordingly
        self.__parse_file()

        hurricane_graph: Graph = Graph(
            undirected=True,
            equip_time=self.__args["Q"],
            unequip_time=self.__args["U"],
            kit_slower=self.__args["P"]
        )

        for node_id, node_params in self.__nodes.items():
            hurricane_graph.add_node(
                node_id,
                has_kit="K" in node_params,
                num_people=next((int(param[1:]) for param in node_params if param.startswith("P")), 0)
            )

        for edge_id, edge_params in self.__edges.items():
            hurricane_graph.add_edge(
                edge_params[0],
                edge_params[1],
                edge_id=edge_id,
                weight=edge_params[2],
                flooded=edge_params[3],
            )

        return hurricane_graph

    def __parse_file(self) -> None:
        """ manages the context of file reading and line parsing """
        with open(self.__file_path) as file:
            for line in file:
                line: str = line.strip()

                # we assume that all line parameters starts with '#', every other prefix line will be skipped
                if not line or not line.startswith("#"):
                    continue

                self.__parse_line(line)

    def __parse_line(self, line: str) -> None:
        """
        parses a single line in the ASCII file that represents the graph parameters

        Args:
            line (str): a stripped line from the file
        """
        # remove comments
        clean_line: str = line.split(";", 1)[0].strip()
        line_arguments: list[str] = clean_line.split()

        # in every line, the first token is always the prefix glued to the parameter (e.g.,: #N)
        tag: str = line_arguments[0]

        if tag[1] in ["N", "U", "Q", "P"]:
            self.__args[tag[1]] = int(line_arguments[1])
        elif tag.startswith("#V"):
            self.__nodes[int(tag[2:])] = line_arguments[1:]
        elif tag.startswith("#E"):
            self.__edges[int(tag[2:])] = (
                int(line_arguments[1]),
                int(line_arguments[2]),
                int(line_arguments[3][1:]),
                "F" in line_arguments
            )
