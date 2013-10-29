Well Rounded Lattices in the Plane
=====

Python code to study the Zeta function on well-rounded lattices and create MatLab style plots (matplotlib).

Uses numpy to crunch numbers and ruby to do some file renaming.

**contributors** Jonathan Raiman

**special thanks** Lenny Fukshansky (Thesis advisor).

![Image](../master/zeta_function?raw=true)

Use
----

In your python REPL run

	import zeta_function.py as zp

and then you can access functions such as:

	zp.compare_det_lattices(1, 10)

to obtain the zeta function for that particular lattice.

You can also do:

	zp.plot_zeta_function(1, 10)

to get a MatLab styled matplotlib output of the zeta function over different minimal norms and different Zeta function exponents.

Finally a randomized function can also be useful:

	zp.random_zeta_function(2, 1000) # or 10,000,000?
