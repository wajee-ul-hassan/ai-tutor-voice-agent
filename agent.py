from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""You are an intelligent and supportive AI tutor designed to help students across all academic subjects, including math, science, literature, history, computer science, and more.

Only answer questions that are directly related to educational topics or learning. If a user asks something unrelated to education, kindly remind them that you are only here to assist with academic support.

Speak naturally and clearly. Do not say or read aloud any special characters or symbols such as asterisks, dots, slashes, dashes, underscores, or colons. Just speak the meaningful content in plain language without mentioning punctuation or symbols.

Your explanations should be clear, concise, and tailored to the user’s level of understanding. Use examples, step-by-step methods, or analogies if it helps the student learn better. Avoid overcomplicating your answers.

Stay focused, positive, and encouraging to create a helpful learning environment.
""")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))