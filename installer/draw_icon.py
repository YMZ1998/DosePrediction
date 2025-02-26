import matplotlib.pyplot as plt
from PIL import Image

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.axis('off')

ax.text(0, 0, 'DP', fontsize=170, fontweight='bold', va='center', ha='center', alpha=0.8, color='blue')
# ax.text(0, -0.5, 'Prediction', fontsize=70, fontweight='bold', va='center', ha='center', alpha=0.8, color='blue')

plt.savefig('DP_logo.png', transparent=False, bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()

png_image = Image.open('DP_logo.png')
print(png_image.size)
high_res_sizes = [(256, 256)]
png_image.save('DP_logo.ico', format='ICO', sizes=high_res_sizes)

png_image = Image.open('DP_logo.ico')
print(png_image.size)


