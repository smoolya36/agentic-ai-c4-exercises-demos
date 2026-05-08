Reflection Report: Multi-Agent Inventory & Sales System
Overview
In this project, I designed and implemented a multi-agent system to handle inventory, quoting, and order fulfillment for a paper supply business. The goal was to model a realistic sales workflow where responsibility is distributed across specialized agents and coordinated by a central orchestrator agent. The system integrates structured database tools with agent reasoning to process customer requests end-to-end.

Multi-Agent Architecture and Workflow
The system consists of four primary agents:

MunderDifflinOrchestrator
InventoryAgent
QuotingAgent
OrderingAgent

The orchestrator agent acts as the single entry point for customer requests. Rather than directly performing business logic, it delegates tasks to managed agents, allowing each agent to focus on a single responsibility:

The InventoryAgent checks stock availability and identifies shortages.
The QuotingAgent generates customer quotes using pricing rules and historical quotes.
The OrderingAgent fulfills orders when inventory is available or places restock orders when inventory is insufficient.

Each agent interacts with the system through explicitly defined tools, such as:

check_inventory_for_item
check_inventory_for_request
lookup_quote_history
generate_customer_quote
place_restock_order
fulfill_customer_order
get_supplier_delivery_date

This separation of concerns makes the workflow easier to reason about and mirrors how real-world organizations split inventory control, sales, and fulfillment responsibilities.

Tool-Driven Design Decisions
Each tool was designed to represent a clear business operation and to safely handle natural-language input from agents. During development, several practical considerations influenced the design:

Tools were made defensive (e.g., handling None values for optional parameters like markup).
Item names are normalized to handle variations such as “reams of paper” vs. catalog names like “Standard copy paper”.
Tools return structured responses instead of raising exceptions, allowing agents to recover and adapt.

This approach significantly improved system robustness when interacting with LLM-generated inputs.

Testing and Observations
The system was tested using a sequence of customer quote requests across multiple dates. Observed behaviors include:





















ScenarioOutcomeSufficient inventory availableOrder fulfilled immediatelyPartial inventory shortageRestock order placed, ETA returnedUnsupported item request (e.g. A3 paper)Graceful rejection with explanation
These tests confirmed that:

Inventory levels were correctly updated in the database.
Cash balances and transactions were recorded accurately.
Quotes and fulfillment decisions reflected real inventory constraints.


Trade-offs and Design Choices
Several trade-offs were made intentionally:


Simplicity vs. Extensibility
The workflow prioritizes clarity and correctness over highly dynamic agent behaviors. While more agents or tools could be added, the current design keeps the orchestration logic understandable and reviewer-verifiable.


Deterministic logic vs. full autonomy
While agents use LLM reasoning, critical business logic (pricing, inventory updates, transaction recording) remains deterministic. This reduces unpredictability and ensures reproducible results.


Explicit tools vs. implicit reasoning
Exposing well-defined tools allows precise control over database interactions and minimizes accidental or unsafe operations by agents.


These choices were made to balance realism, reliability, and evaluation clarity.

Scalability Considerations
If the system were to scale further, several extensions would be straightforward:


Additional agents
For example, a CustomerSupportAgent or AnalyticsAgent could be added without modifying existing agents.


Larger inventories and datasets
The database-driven design supports larger inventories and longer transaction histories with minimal changes.


Increased request volume
The orchestrator-based architecture allows parallel processing or queue-based handling in future versions.


More sophisticated pricing logic
Bulk discounts, customer tiers, or promotional pricing could be integrated as additional tools or agent rules.



Key Learnings
This project reinforced the importance of:

Designing agents with clear responsibilities
Using explicit tools to ground LLM reasoning in structured operations
Ensuring systems are robust to imperfect language input
Treating the orchestrator as a true coordinator rather than a manual controller

Overall, the system demonstrates how multi-agent architectures can effectively manage complex workflows while remaining modular, readable, and extensible.