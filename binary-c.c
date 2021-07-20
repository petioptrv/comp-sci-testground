#include <stdio.h>

int sign(int a);
int myabs(int a);

int main(int argc, char *argv[]) {
	printf("sign(2): %d\n", sign(2));
	printf("sign(-2): %d\n", sign(-2));
	printf("myabs(2): %d\n", myabs(2));
	printf("myabs(-2): %d\n", myabs(-2));
}

int sign(int a) {
	return - (a >> 31);
}

int myabs(int a) {
	int high_bit_mask = a >> 31;
	return (a ^ high_bit_mask) - high_bit_mask;
}