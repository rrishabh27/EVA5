import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        # self.load_state_dict(parameters)

        self.load_state_dict(parameters, strict=False) # if there are additional params not required by the current model in weight file
