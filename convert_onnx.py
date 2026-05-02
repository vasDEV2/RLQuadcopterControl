import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self):
        # MUST exactly match the trained rl_games actor architecture
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)
        # self.fc5 = nn.Linear(8, 4)

        self.act = nn.ELU()
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        # x = self.act(self.fc4(x))
        x = self.fc4(x)
        return x
    




def convert_network():
    # ------------------------------------------------------------
    # Load rl_games PPO checkpoint
    # ------------------------------------------------------------
    checkpoint = torch.load("02_only1.pth", map_location="cpu", weights_only=False)
    # print(checkpoint.keys())
    # rl_games stores weights under "model"
    model_state_dict = checkpoint["model"]
    # print(model_state_dict)

    print(model_state_dict.keys())
    # print(model_state_dict["a2c_network.actor_mlp.4.weight"].shape)
    # print(model_state_dict['a2c_network.adaption_network.linear_layer.4.weight'].shape)

    # ------------------------------------------------------------
    # Map rl_games keys → our ActorNetwork keys
    # ------------------------------------------------------------
    mapped_state_dict = {
        "fc1.weight": model_state_dict["_orig_mod.a2c_network.actor_mlp.0.weight"],
        "fc1.bias":   model_state_dict["_orig_mod.a2c_network.actor_mlp.0.bias"],
        "fc2.weight": model_state_dict["_orig_mod.a2c_network.actor_mlp.2.weight"],
        "fc2.bias":   model_state_dict["_orig_mod.a2c_network.actor_mlp.2.bias"],
        "fc3.weight": model_state_dict["_orig_mod.a2c_network.actor_mlp.4.weight"],
        "fc3.bias":   model_state_dict["_orig_mod.a2c_network.actor_mlp.4.bias"],
        # "fc4.weight": model_state_dict["a2c_network.actor_mlp.6.weight"],
        # "fc4.bias":   model_state_dict["a2c_network.actor_mlp.6.bias"],
        "fc4.weight": model_state_dict["_orig_mod.a2c_network.mu.weight"],
        "fc4.bias":   model_state_dict["_orig_mod.a2c_network.mu.bias"]
    }

    # ------------------------------------------------------------
    # Initialize and load model
    # ------------------------------------------------------------
    actor_model = ActorNetwork()
    actor_model.load_state_dict(mapped_state_dict, strict=True)
    actor_model.eval()

    # ------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------
    sample_input = torch.randn(1, 16)
    with torch.no_grad():
        pytorch_output = actor_model(sample_input)

    print("PyTorch output shape:", pytorch_output.shape)

    # ------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------
    torch.onnx.export(
        actor_model,
        sample_input,
        "test_rigour_2.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actor_latent"],
        dynamic_axes={
            "obs": {0: "batch"},
            "actor_latent": {0: "batch"},
        },
    )

    print("✅ Exported ONNX model: gen_ppo_actor.onnx")


if __name__ == "__main__":
    convert_network()
