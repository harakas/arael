set terminal pngcairo enhanced size 1600, 640 font "sans,26" background rgb "white"
set output 'capper.png'

set grid lw 1.2 lc rgb "#d0d0d0"
set border lw 1.5

set xrange [-10:10]

gamma = 2 * sqrt(25) / pi
a(x) = gamma * atan(x / gamma)

set multiplot layout 1, 2

set key bottom right
set xtics 5
set ytics 5
set ylabel ""
set title "{/sans=28{/Symbol a}(x) vs x}"

plot \
    x         with lines lw 3 lc rgb "#7e57c2" title "x", \
    a(x)      with lines lw 3 lc rgb "#26a69a" title "{/Symbol a}(x)"

set key center top
set yrange [0:30]
set ytics 5
set ytics add ("{/Symbol D}S_{max}" 25)
set title "{/sans=28{/Symbol a}(x)^2 vs x^2}"

plot \
    x**2      with lines lw 3 lc rgb "#7e57c2" title "x^2", \
    a(x)**2   with lines lw 3 lc rgb "#26a69a" title "{/Symbol a}(x)^2"

unset multiplot
