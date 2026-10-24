from uagents import Agent

from protocols import travel_protocol

agent = Agent()

agent.include(travel_protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
