"""
Débat entre écrivains — LangGraph + LangChain + Ollama

Trois écrivains historiques (Victor Hugo, Voltaire, Rousseau) débattent
sur un sujet donné, chacun avec son style littéraire propre.

Deux modes :
  - Mode "prompt" : utilise mistral avec des system prompts riches
  - Mode "finetuned" : utilise des modèles fine-tunés par auteur (voir finetune/)

Concepts LangGraph :
  - StateGraph, TypedDict, reducers (operator.add)
  - Nodes, edges, conditional edges
  - stream() pour affichage progressif

Concepts LangChain :
  - ChatOllama : modèle local
  - SystemMessage / HumanMessage / AIMessage
"""

import operator
import sys
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. STATE
# ---------------------------------------------------------------------------
class DebateState(TypedDict):
    topic: str
    messages: Annotated[list, operator.add]
    round: int
    max_rounds: int


# ---------------------------------------------------------------------------
# 2. ÉCRIVAINS — Personnalités basées sur leurs oeuvres réelles
# ---------------------------------------------------------------------------
WRITERS = {
    "hugo": {
        "name": "Victor Hugo",
        "model": "hugo-writer",
        "system_prompt": (
            "Tu es Victor Hugo, écrivain romantique français du XIXe siècle. "
            "Tu t'exprimes avec lyrisme, passion et grandeur. Tu défends les opprimés, "
            "tu crois au progrès de l'humanité et à la force de l'amour et de la justice. "
            "Tu utilises des métaphores puissantes et un style emphatique.\n\n"
            "Voici des extraits de tes oeuvres pour t'inspirer :\n\n"
            "« La liberté commence où l'ignorance finit. »\n"
            "« Ceux qui vivent, ce sont ceux qui luttent. »\n"
            "« La mélancolie, c'est le bonheur d'être triste. »\n"
            "« Vous voulez les misérables secourus, moi je veux la misère supprimée. »\n"
            "« La popularité ? C'est la gloire en gros sous. »\n\n"
            "Argumente avec passion et conviction, en 3-5 phrases. Parle en français."
        ),
    },
    "voltaire": {
        "name": "Voltaire",
        "model": "voltaire-writer",
        "system_prompt": (
            "Tu es Voltaire, philosophe et écrivain des Lumières. Tu es sarcastique, "
            "rationnel et mordant. Tu combats les superstitions, l'intolérance et l'injustice "
            "par l'ironie et la raison. Tu as un esprit vif et tu n'hésites pas à provoquer.\n\n"
            "Voici des extraits de tes oeuvres pour t'inspirer :\n\n"
            "« Le doute est inconfortable, la certitude est ridicule. »\n"
            "« Il faut cultiver notre jardin. » — Candide\n"
            "« J'ai décidé d'être heureux parce que c'est bon pour la santé. »\n"
            "« Le secret d'ennuyer est celui de tout dire. »\n"
            "« Ceux qui peuvent vous faire croire des absurdités peuvent vous faire "
            "commettre des atrocités. »\n\n"
            "Argumente avec ironie et lucidité, en 3-5 phrases. Parle en français."
        ),
    },
    "rousseau": {
        "name": "Jean-Jacques Rousseau",
        "model": "rousseau-writer",
        "system_prompt": (
            "Tu es Jean-Jacques Rousseau, philosophe genevois du XVIIIe siècle. "
            "Tu es introspectif, sensible et défenseur de la nature contre la corruption "
            "de la civilisation. Tu prônes le retour à l'authenticité, le contrat social "
            "et l'éducation naturelle. Tu es parfois mélancolique mais toujours sincère.\n\n"
            "Voici des extraits de tes oeuvres pour t'inspirer :\n\n"
            "« L'homme est né libre, et partout il est dans les fers. » — Du contrat social\n"
            "« La nature a fait l'homme heureux et bon, mais la société le déprave. »\n"
            "« Vivre, ce n'est pas respirer, c'est agir. »\n"
            "« La patience est amère, mais son fruit est doux. »\n"
            "« Je sens mon coeur et je connais les hommes. » — Les Confessions\n\n"
            "Argumente avec sincérité et profondeur, en 3-5 phrases. Parle en français."
        ),
    },
}

MODERATOR_PROMPT = (
    "Tu es un modérateur littéraire cultivé. Tu animes un débat entre Victor Hugo, "
    "Voltaire et Jean-Jacques Rousseau. Tu résumes les arguments de chacun avec finesse "
    "et relances le débat. 2-3 phrases max. Parle en français."
)

WRITER_ORDER = ["hugo", "voltaire", "rousseau"]


# ---------------------------------------------------------------------------
# 3. NODES
# ---------------------------------------------------------------------------
def make_writer_node(writer_id: str):
    """Crée un noeud LangGraph pour un écrivain donné."""
    writer = WRITERS[writer_id]
    llm = ChatOllama(model=writer["model"], temperature=0.8)

    def node_fn(state: DebateState) -> dict:
        messages = [
            SystemMessage(content=writer["system_prompt"]),
            HumanMessage(content=f"Le sujet du débat est : {state['topic']}"),
        ]
        for msg in state["messages"]:
            messages.append(msg)
        messages.append(
            HumanMessage(
                content=f"C'est ton tour, {writer['name']}. Donne ton argument sur ce sujet."
            )
        )

        response = llm.invoke(messages)

        return {
            "messages": [
                AIMessage(
                    content=f"[{writer['name']}] {response.content}",
                    name=writer_id,
                )
            ],
        }

    node_fn.__name__ = f"node_{writer_id}"
    return node_fn


def moderator_node(state: DebateState) -> dict:
    """Le modérateur résume le tour et relance."""
    llm = ChatOllama(model="mistral", temperature=0.5)
    current_round = state["round"]

    messages = [
        SystemMessage(content=MODERATOR_PROMPT),
        HumanMessage(content=f"Sujet : {state['topic']}"),
    ]
    for msg in state["messages"][-6:]:
        messages.append(msg)
    messages.append(
        HumanMessage(content=f"Résume ce tour {current_round + 1} et relance le débat.")
    )

    response = llm.invoke(messages)

    return {
        "messages": [
            AIMessage(content=f"[Modérateur] {response.content}", name="moderateur")
        ],
        "round": current_round + 1,
    }


# ---------------------------------------------------------------------------
# 4. ROUTING
# ---------------------------------------------------------------------------
def route_after_moderator(state: DebateState) -> Literal["hugo", "__end__"]:
    if state["round"] >= state["max_rounds"]:
        return END
    return WRITER_ORDER[0]


# ---------------------------------------------------------------------------
# 5. GRAPH
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(DebateState)

    for writer_id in WRITER_ORDER:
        builder.add_node(writer_id, make_writer_node(writer_id))
    builder.add_node("moderateur", moderator_node)

    builder.add_edge(START, WRITER_ORDER[0])
    for i, writer_id in enumerate(WRITER_ORDER):
        if i < len(WRITER_ORDER) - 1:
            builder.add_edge(writer_id, WRITER_ORDER[i + 1])
        else:
            builder.add_edge(writer_id, "moderateur")

    builder.add_conditional_edges(
        "moderateur",
        route_after_moderator,
        {WRITER_ORDER[0]: WRITER_ORDER[0], END: END},
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  DÉBAT LITTÉRAIRE — Hugo vs Voltaire vs Rousseau")
    print("  LangGraph + LangChain + Ollama (local)")
    print("=" * 60)

    topic = input("\nSujet du débat : ").strip()
    if not topic:
        topic = "Le progrès technique rend-il l'homme plus heureux ?"

    max_rounds = int(input("Nombre de tours (défaut 2) : ").strip() or "2")

    print(f"\nSujet : {topic}")
    print(f"Tours : {max_rounds}")
    print(f"Écrivains : {', '.join(WRITERS[w]['name'] for w in WRITER_ORDER)}")
    print("-" * 60)

    graph = build_graph()

    initial_state = {
        "topic": topic,
        "messages": [],
        "round": 0,
        "max_rounds": max_rounds,
    }

    for chunk in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    print(f"\n{msg.content}")
                    print("-" * 40)

    print("\n" + "=" * 60)
    print("  Débat terminé !")
    print("=" * 60)


if __name__ == "__main__":
    main()
