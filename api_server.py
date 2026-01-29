from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.Legal_Research.act_case_fusion_agent import fuse_bare_act_and_case_law

app = FastAPI(title="Nyaymalaw API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    issue: str

@app.post("/search")
def search_law(query: Query):
    result = fuse_bare_act_and_case_law.run(issue=query.issue)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
