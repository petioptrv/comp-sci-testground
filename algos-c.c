#include <stdio.h>

int binary_search(int target, int numbers[], int size);
int binary_search_recurse(int target, int numbers[], int start, int end);

int main(int argc, char *argv[]) {
	// --- BINARY SEARCH ---
	int numbers[] = {0,1,2,3,4};
	printf("%d\n", binary_search(5, numbers, 5));
	printf("%d\n", binary_search_recurse(5, numbers, 0, 5));
}

int binary_search(int target, int numbers[], int size) {
	int s = 0;
	int e = size;
	int pole;
	int t_idx = -1;
	
	while (s < e && numbers[pole] != target) {
		pole = (s + e) / 2;
		if (target < numbers[pole]) {
			e = pole;
		} else {
			s = pole + 1;
		}
	}
	
	if (numbers[pole] == target) {
		t_idx = pole;
	}
	
	return t_idx;
}

int binary_search_recurse(int target, int numbers[], int start, int end) {
	int pole = (start + end) / 2;
	int res = -1;
	
	if (start < end) {
		if (numbers[pole] < target) {
			res = binary_search_recurse(target, numbers, pole + 1, end);
		} else if (numbers[pole] > target) {
			res = binary_search_recurse(target, numbers, start, pole);
		} else {
			res = pole;
		}
	}
	
	return res;
}
