import numpy as np

nums = [0, 1, 2, 3, 4]

print(nums)
print(nums[2:4])
print(nums[2:])
print(nums[:2])
print(nums[:])
print(nums[:-1])
nums[2:4] = [8, 9]
print(nums)

a = np.array([1, 2, 3, 4, 5])

print(a[1:3])
print(a[-1])
a[0:2] = 9
print(a)

b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(b[:, 1])
print(b[-1])
print(b[-1, :])
print(b[-1, ...])
print(b[0:2, :])
