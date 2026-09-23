from agents.dbc import DBCAgent
from agents.dppo import DPPOAgent
from agents.fbc import FBCAgent
from agents.resip import ResiPAgent

agents = dict(
    dbc=DBCAgent,
    dppo=DPPOAgent,
    fbc=FBCAgent,
    resip=ResiPAgent,
)
