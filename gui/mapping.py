import cv2

#when shit hits the fan, this is the script to run.

############### INIT ###############

grid = cv2.imread("empty_grid.png")

block_size = 68

colors = [
	(0,0,255), #red for large colony
	(0,0,255),
	(0,255,255), #yellow for coral fragmants
	(0,255,255),
	(255,0,0), #blue for sea stars
	(255,0,0),
	(0,255,0) # green for sponge
] # grid cirles colors

poi = [
	"large colony 1",
	"large colony 2",
	"coral fragmant 1",
	"coral fragmant 2",
	"sea star 1",
	"sea star 2",
	"sponge"
] #points of interest

############### Fx ###############

def draw(block, color):

	x = (block*block_size) - int(block_size/2)

	if block <= 9:
		y = int(block_size/2) -10
		x = (block*block_size) - int(block_size/2)
	elif block > 9 and block < 19:
		y = (block_size*2) - int(block_size/2)-10
		x = ((block-9)*block_size) - int(block_size/2)
	elif block >= 19 and block <= 27:
		y = (block_size*3) - int(block_size/2)-10
		x = ((block-18)*block_size) - int(block_size/2)
	else:
		print("error getting x or y")

	pos = (x, y)
	radius = int(block_size/2 - 10)
	
	cv2.circle(grid, pos, radius, color, 3)

def get_block(i):

	while(True):
		try:

			block = input("enter block number for "+poi[i]+" : ")
			block = int(block)

			if block > 27 or block < 1:
				raise ValueError

		except ValueError:
			print("this is not an appropriate entry, please try again.")

		except Exception as e:
			print("NEW ERROR :"+e)

		else:
			break
	return block

def print_head():
	print("""
					PLAN B
	----------------------------------------
	-the grid is a 9x3 grid that is counted 
		from the top left to the bottom right

	-input each corresponding block no. to the item inside

	-starting at first block first row is 1
	-first block middle row is 10
	-first block last row is 19
	----------------------------------------
	""")

############### Main ###############

print_head()

for i in range(7):

	block = get_block(i)
	draw(block, colors[i])


cv2.imshow("grid", grid)
cv2.imwrite("grid.png", grid)

cv2.waitKey(0)
cv2.destroyAllWindows()