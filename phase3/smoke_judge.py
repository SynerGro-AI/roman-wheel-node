"""One tiny xAI judge call to confirm the interface works (cost: ~150 tokens). Never prints the key."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from factcheck.interfaces import Passage, XAIJudge

j = XAIJudge(model=sys.argv[1] if len(sys.argv) > 1 else "grok-4.20-0309-non-reasoning")
p = j.score("Aspirin irreversibly inhibits cyclooxygenase.",
            [Passage("demo", "Aspirin acetylates a serine residue of COX-1 and COX-2, irreversibly inhibiting them.")])
q = j.score("Aspirin activates cyclooxygenase.",
            [Passage("demo", "Aspirin acetylates a serine residue of COX-1 and COX-2, irreversibly inhibiting them.")])
print({"model": j.model, "p_supported_true_claim": p, "p_supported_false_claim": q, "usage_last": j.last_usage})
