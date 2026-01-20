import numpy as np

act = np.array([0.7051835060119629, 0.8948826193809509, 0.18240785598754883, 0.5674643516540527])

allocation_matrix = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, 0.13, 0.13, -0.13],
            [-0.13, 0.13, -0.13, 0.13],
            [-0.025, 0.025, -0.025, 0.025],
        ])

inverse = np.linalg.pinv(allocation_matrix)


wrench = allocation_matrix @ act.T

print(wrench)

print(f"WRENCH: {wrench}")
print(f"Inverse: {inverse}")

forces = inverse @ wrench.T

print(f"inveretd foprces {forces}")