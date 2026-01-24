import asyncio
from ollama import chat, AsyncClient


SYSTEM_PROMPT = """
You are giving instructions to a robot.
Available instructions are: "go to", "pick up", "toggle", "go to goal".
Available objects are: "key", "door", "ball", "box".
Available colors are: "red", "green", "yellow", "blue", "purple", "grey".

Give a simple answer, consisting only of available instructions, colors and objects.
Don't look for alternative routes.
Give your answer in a step by step format. Try to have as little as possible steps in your answer.
If the door is locked, a key of the same color is needed. Doors are unlocked by toogling them while holding the key.
Keys can only toggle locked doors of the same color. Robot must first pick up a key before toggling the door.
Boxes do not need keys to be toggled.
Consider everything directly accessible unless it is in another room.
Box can contain other objects with the same color. If the object you are looking for is not present but there is a box, try opening the box.
Make sure to reference only objects present in the scene.
Do not tell the robot to go into the room or to open the unlocked door.

Examples:

The scene contains:
- green ball
- yellow ball
- blue key
- purple door
- yellow door
- green key
- goal
Mission: go to goal'

Answer:
1. go to goal

The scene contains:
- green ball
- purple ball
- red box
- purple box
- green box
- blue key
Mission: pick up purple box'

Answer:
1. pick up purple box

The scene contains:
Two rooms. Left and right.
There is a locked grey door between the rooms
Left room contains:
- goal
- green ball
- yellow key
Right room contains:
- robot
- grey box
- red key
- grey ball
Mission: go to goal

Answer:
1. toggle grey box
2. pick up grey key
3. toggle grey door
4. go to goal
"""

env_mission_gtg_0 = """The scene contains:
Only one room.
- yellow key
- grey ball
- green box
- red box
- purple door
- grey box
- goal
Mission: go to goal"""

env_mission_gtg_1 = """The scene contains:
Only one room.
- red ball
- green ball
- red key
- grey box
- purple box
- red door
- goal
Mission: go to goal"""

env_mission_gto_0 = """The scene contains:
Only one room.
- green door
- yellow ball
- green key
- grey ball
- green box
- red box
Mission: go to green key"""

env_mission_gto_1 = """The scene contains:
Only one room.
- purple door
- purple key
- grey key
- yellow ball
- purple ball
- grey ball
Mission: go to grey ball"""

env_mission_pkp_0 = """The scene contains:
Only one room.
- green key
- grey ball
- blue box
- yellow key
- grey box
- blue key
Mission: pick up blue key"""

env_mission_pkp_1 = """The scene contains:
Only one room.
- blue key
- red ball
- purple box
- red box
- purple key
- blue box
Mission: pick up purple box"""

env_mission_opn_0 = """The scene contains:
Only one room.
- purple door
- blue box
- yellow door
- red door
- yellow box
- grey box
Mission: toggle blue box"""

env_mission_opn_1 = """The scene contains:
Only one room.
- green door
- grey box
- yellow door
- red box
- blue box
- red door
Mission: toggle green door"""

env_mission_multi_0 = """The scene contains:
Two rooms. Left and right.
There is an unlocked green door between the rooms
Left room contains:
- goal
- blue box
- purple box
Right room contains:
- robot
- red key
- grey ball
- purple ball
Mission: go to goal"""

env_mission_multi_1 = """The scene contains:
Two rooms. Left and right.
There is an unlocked purple door between the rooms
Left room contains:
- goal
- yellow key
- red box
Right room contains:
- robot
- green key
- grey ball
- yellow box
Mission: go to goal"""

env_mission_multi_2 = """The scene contains:
Three rooms. Upper left, lower left and right.
There is an unlocked red door between the upper left and lower left rooms.
There is a locked yellow door between the upper left and right rooms.
There is a locked grey door between the lower left and right rooms.
Upper left room contains:
- goal
Lower left room contains:
Right room contains:
- robot
- grey box
- yellow key
- yellow ball
Mission: go to goal"""

env_mission_multi_3 = """The scene contains:
Three rooms. Upper left, lower left and right.
There is an unlocked yellow door between the upper left and lower left rooms.
There is an unlocked blue door between the upper left and right rooms.
There is an unlocked purple door between the lower left and right rooms.
Upper left room contains:
- yellow ball
Lower left room contains:
- grey box
Right room contains:
- robot
- goal
- green key
- blue ball
Mission: go to goal"""

env_mission_multi_4 = """The scene contains:
Four rooms. Upper left, lower left, upper right and lower right.
There is a locked red door between the upper left and lower left rooms.
There is an unlocked yellow door between the upper right and lower right rooms.
There is a locked green door between the upper left and upper right rooms.
There is a locked purple door between the lower left and lower right rooms.
Upper left room contains:
- robot
- green box
- red box
Lower left room contains:
- goal
- purple key
Upper right room contains:
- red ball
Lower right room contains:
- grey key
- blue box
Mission: go to goal"""

env_mission_multi_5 = """The scene contains:
Four rooms. Upper left, lower left, upper right and lower right.
There is a locked grey door between the upper left and lower left rooms.
There is a locked purple door between the upper right and lower right rooms.
There is a locked blue door between the upper left and upper right rooms.
There is a locked yellow door between the lower left and lower right rooms.
Upper left room contains:
- purple ball
Lower left room contains:
- grey key
- red ball
Upper right room contains:
- blue key
Lower right room contains:
- robot
- yellow box
- purple box
- goal
Mission: go to goal"""

# copy-paste na deep seek chat
env_mission_multi_6 = """The scene contains:
Four rooms. Upper left, lower left, upper right and lower right.
There is a locked grey door between the upper left and lower left rooms.
There is a locked purple door between the upper right and lower right rooms.
There is a locked blue door between the upper left and upper right rooms.
There is a locked yellow door between the lower left and lower right rooms.
Upper left room contains:
- purple ball
- goal
Lower left room contains:
- grey key
- red ball
Upper right room contains:
- blue key
Lower right room contains:
- robot
- yellow box
- purple box
Mission: go to goal"""


client = AsyncClient(host='http://127.0.0.1:11434')

async def chat():
    async for part in await client.chat(
        model='qwen3:30b',
        options={
            'seed': 42,
            'num_ctx': 64000,
            'repeat_last_n': -1,
            'repeat_penalty': 2.0,
            'temperature': 0.0,
            'top_k': 20,
            'top_p': 0.95,
            'min_p': 0.
        },
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': "".join(env_mission_multi_6)
            }
        ],
        stream=True
    ):
        print(part['message']['content'], end='', flush=True)

asyncio.run(chat())
print()