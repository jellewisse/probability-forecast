from .base import MixtureModel, MixtureMember
from scipy.stats import norm

class GaussianMixture(MixtureModel):
    """"""

    def __init__(self, member_count):

        members = self._generate_members(member_count)
        super.__self__(members)

    def _generate_members(self, member_count):

        members = []
        for i in range(0, member_count):
            parameters = {}
            member = MixtureMember(norm, )
            members.append(member)
        pass
