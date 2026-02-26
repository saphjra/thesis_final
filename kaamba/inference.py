context = initial_gaze

for t in range(pred_steps):
    pred = model(image, context)
    next_gaze = pred[:, -1]
    context = torch.cat([context, next_gaze.unsqueeze(1)], dim=1)