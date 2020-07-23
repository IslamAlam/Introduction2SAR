FUNCTION rect, x
    abs_x = abs( x )
    answer = x

    g = where( abs_x ge 0.5, count_g)
    l = where( abs_x le 0.5, count_l )
    if (count_g ne 0) then answer(g)=0.0
    if (count_l ne 0) then answer(l)=1.0

    RETURN, answer
END