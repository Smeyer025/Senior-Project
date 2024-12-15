/**
 * OutputContainer Component: Container for output
 */

import "./OutputContainer.css";

/**
 * OutputContainer()
 * 
 * NAME
 *    OutputContainer() - Handles layout for output from analysis
 * 
 * SYNOPSIS
 *    React.JSX.Element OutputContainer(a_props)
 *      a_props --> properties
 *          a_props.children --> Text to be shown in OutputContainer
 * 
 * DESCRIPTION
 *    This function creates a container for output
 * 
 * RETURNS 
 *    Returns jsx element for output container
 */
export default function OutputContainer(a_props) {
    const determineColor = (a_children) => {
        console.log(`children: ${a_children}`);
        if (a_children == "positive") {
            return "green";
        } else if (a_children == "negative") {
            return 'red';
        } else {
            return 'black';
        }
    };

    return (
        <div className="OC">
            <h1 style={{color: determineColor(a_props.children)}}>{a_props.children}</h1>
        </div>
    );
};