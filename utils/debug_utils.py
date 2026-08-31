import torch


Anchor_C = None


def check_anchor_modified(pc):
    old_anchor = pc._anchor.data.cpu().clone()

    def checker():
        if torch.isnan(pc._anchor.data.cpu()).any() or (not torch.equal(pc._anchor.data.cpu(), old_anchor)):
            print("⚠️ WARNING: self._anchor has been modified!")
            import traceback
            traceback.print_stack()
            return True
        return False
    
    return checker